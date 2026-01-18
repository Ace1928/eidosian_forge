from typing import List
from vllm.utils import Device
def append_tokens(self, token_ids: List[int]) -> None:
    assert len(token_ids) <= self.get_num_empty_slots()
    curr_idx = self.num_tokens
    self.token_ids[curr_idx:curr_idx + len(token_ids)] = token_ids
    self.num_tokens += len(token_ids)