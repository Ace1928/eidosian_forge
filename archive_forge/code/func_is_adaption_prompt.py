from collections import namedtuple
from dataclasses import dataclass, field
from peft.config import PeftConfig
from peft.utils import PeftType
from .utils import llama_compute_query_states
@property
def is_adaption_prompt(self) -> bool:
    """Return True if this is an adaption prompt config."""
    return True