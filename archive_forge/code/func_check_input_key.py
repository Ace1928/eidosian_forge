import warnings
from typing import Any, Dict, List, Set
from langchain_core.memory import BaseMemory
from langchain_core.pydantic_v1 import validator
from langchain.memory.chat_memory import BaseChatMemory
@validator('memories')
def check_input_key(cls, value: List[BaseMemory]) -> List[BaseMemory]:
    """Check that if memories are of type BaseChatMemory that input keys exist."""
    for val in value:
        if isinstance(val, BaseChatMemory):
            if val.input_key is None:
                warnings.warn(f'When using CombinedMemory, input keys should be so the input is known.  Was not set on {val}')
    return value