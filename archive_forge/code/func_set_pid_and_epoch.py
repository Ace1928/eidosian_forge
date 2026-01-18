from enum import Enum
from collections import namedtuple, defaultdict, deque
from aiokafka.structs import TopicPartition
from aiokafka.util import create_future
def set_pid_and_epoch(self, pid: int, epoch: int):
    self._pid_and_epoch = PidAndEpoch(pid, epoch)
    self._pid_waiter.set_result(None)
    if self.transactional_id:
        self._transition_to(TransactionState.READY)