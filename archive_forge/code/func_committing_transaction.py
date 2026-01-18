from enum import Enum
from collections import namedtuple, defaultdict, deque
from aiokafka.structs import TopicPartition
from aiokafka.util import create_future
def committing_transaction(self):
    if self.state == TransactionState.ABORTABLE_ERROR:
        self._transaction_waiter.result()
    self._transition_to(TransactionState.COMMITTING_TRANSACTION)
    self.notify_task_waiter()