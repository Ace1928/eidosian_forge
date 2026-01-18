from enum import Enum
from collections import namedtuple, defaultdict, deque
from aiokafka.structs import TopicPartition
from aiokafka.util import create_future
def needs_transaction_commit(self):
    if self.state == TransactionState.COMMITTING_TRANSACTION:
        return TransactionResult.COMMIT
    elif self.state == TransactionState.ABORTING_TRANSACTION:
        return TransactionResult.ABORT
    else:
        return