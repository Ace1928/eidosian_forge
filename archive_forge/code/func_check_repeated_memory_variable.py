import warnings
from typing import Any, Dict, List, Set
from langchain_core.memory import BaseMemory
from langchain_core.pydantic_v1 import validator
from langchain.memory.chat_memory import BaseChatMemory
@validator('memories')
def check_repeated_memory_variable(cls, value: List[BaseMemory]) -> List[BaseMemory]:
    all_variables: Set[str] = set()
    for val in value:
        overlap = all_variables.intersection(val.memory_variables)
        if overlap:
            raise ValueError(f'The same variables {overlap} are found in multiplememory object, which is not allowed by CombinedMemory.')
        all_variables |= set(val.memory_variables)
    return value