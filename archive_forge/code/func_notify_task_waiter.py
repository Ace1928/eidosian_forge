from enum import Enum
from collections import namedtuple, defaultdict, deque
from aiokafka.structs import TopicPartition
from aiokafka.util import create_future
def notify_task_waiter(self):
    if self._task_waiter is not None and (not self._task_waiter.done()):
        self._task_waiter.set_result(None)