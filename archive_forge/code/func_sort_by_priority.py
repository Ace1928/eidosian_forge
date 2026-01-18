from collections import deque
from typing import Deque
from vllm.sequence import SequenceGroup
def sort_by_priority(self, now: float, seq_groups: Deque[SequenceGroup]) -> Deque[SequenceGroup]:
    return deque(sorted(seq_groups, key=lambda seq_group: self.get_priority(now, seq_group), reverse=True))