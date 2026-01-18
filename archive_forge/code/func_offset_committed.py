from enum import Enum
from collections import namedtuple, defaultdict, deque
from aiokafka.structs import TopicPartition
from aiokafka.util import create_future
def offset_committed(self, tp, offset, group_id):
    pending_group_id, pending_offsets, fut = self._pending_txn_offsets[0]
    assert pending_group_id == group_id
    assert tp in pending_offsets and pending_offsets[tp].offset == offset
    del pending_offsets[tp]
    if not pending_offsets:
        fut.set_result(None)
        self._pending_txn_offsets.popleft()