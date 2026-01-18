from enum import Enum
from collections import namedtuple, defaultdict, deque
from aiokafka.structs import TopicPartition
from aiokafka.util import create_future
def maybe_add_partition_to_txn(self, tp: TopicPartition):
    if self.transactional_id is None:
        return
    assert self.is_in_transaction()
    if tp not in self._txn_partitions:
        self._pending_txn_partitions.add(tp)
        self.notify_task_waiter()