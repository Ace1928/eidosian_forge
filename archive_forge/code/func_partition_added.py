from enum import Enum
from collections import namedtuple, defaultdict, deque
from aiokafka.structs import TopicPartition
from aiokafka.util import create_future
def partition_added(self, tp: TopicPartition):
    self._pending_txn_partitions.remove(tp)
    self._txn_partitions.add(tp)