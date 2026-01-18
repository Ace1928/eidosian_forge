from collections import deque
from typing import Deque
from vllm.sequence import SequenceGroup
@classmethod
def get_policy(cls, policy_name: str, **kwargs) -> Policy:
    return cls._POLICY_REGISTRY[policy_name](**kwargs)