from enum import Enum
from collections import namedtuple, defaultdict, deque
from aiokafka.structs import TopicPartition
from aiokafka.util import create_future
def error_transaction(self, exc):
    self._transition_to(TransactionState.ABORTABLE_ERROR)
    self._txn_partitions.clear()
    self._txn_consumer_group = None
    self._pending_txn_partitions.clear()
    for _, _, fut in self._pending_txn_offsets:
        fut.set_exception(exc)
    self._pending_txn_offsets.clear()
    self._transaction_waiter.set_exception(exc)