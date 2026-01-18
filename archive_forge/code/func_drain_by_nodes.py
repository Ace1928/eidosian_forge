import asyncio
import collections
import copy
import time
from aiokafka.errors import (KafkaTimeoutError,
from aiokafka.record.legacy_records import LegacyRecordBatchBuilder
from aiokafka.record.default_records import DefaultRecordBatchBuilder
from aiokafka.structs import RecordMetadata
from aiokafka.util import create_future, get_running_loop
def drain_by_nodes(self, ignore_nodes, muted_partitions=set()):
    """ Group batches by leader to partition nodes. """
    nodes = collections.defaultdict(dict)
    unknown_leaders_exist = False
    for tp in list(self._batches.keys()):
        if tp in muted_partitions:
            continue
        leader = self._cluster.leader_for_partition(tp)
        if leader is None or leader == -1:
            if self._batches[tp][0].expired():
                batch = self._pop_batch(tp)
                if leader is None:
                    err = NotLeaderForPartitionError()
                else:
                    err = LeaderNotAvailableError()
                batch.failure(exception=err)
            unknown_leaders_exist = True
            continue
        elif ignore_nodes and leader in ignore_nodes:
            continue
        batch = self._pop_batch(tp)
        if not batch.is_empty():
            nodes[leader][tp] = batch
        else:
            batch.done_noack()
    if not self._wait_data_future.done():
        self._wait_data_future.set_result(None)
    self._wait_data_future = self._loop.create_future()
    return (nodes, unknown_leaders_exist)