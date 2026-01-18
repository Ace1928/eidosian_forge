from typing import List
from vllm.utils import Device
def get_last_token_id(self) -> int:
    assert self.num_tokens > 0
    return self.token_ids[self.num_tokens - 1]