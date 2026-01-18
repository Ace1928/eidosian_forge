from enum import Enum
from collections import namedtuple, defaultdict, deque
from aiokafka.structs import TopicPartition
from aiokafka.util import create_future
def has_pid(self):
    return self._pid_and_epoch.pid != NO_PRODUCER_ID