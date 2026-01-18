from enum import Enum
from collections import namedtuple, defaultdict, deque
from aiokafka.structs import TopicPartition
from aiokafka.util import create_future
def add_offsets_to_txn(self, offsets, group_id):
    assert self.is_in_transaction()
    assert self.transactional_id
    fut = create_future()
    self._pending_txn_offsets.append((group_id, offsets, fut))
    self.notify_task_waiter()
    return fut