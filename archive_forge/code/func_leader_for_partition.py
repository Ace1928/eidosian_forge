import collections
import copy
import logging
import threading
import time
from concurrent.futures import Future
from aiokafka import errors as Errors
from aiokafka.conn import collect_hosts
from aiokafka.structs import BrokerMetadata, PartitionMetadata, TopicPartition
def leader_for_partition(self, partition):
    """Return node_id of leader, -1 unavailable, None if unknown."""
    if partition.topic not in self._partitions:
        return None
    elif partition.partition not in self._partitions[partition.topic]:
        return None
    return self._partitions[partition.topic][partition.partition].leader