from enum import Enum
from collections import namedtuple, defaultdict, deque
from aiokafka.structs import TopicPartition
from aiokafka.util import create_future
def consumer_group_added(self, group_id):
    self._txn_consumer_group = group_id