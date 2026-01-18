import numpy as np  # Assuming NumPy is used for efficient array manipulation
import random
import types
import importlib.util
import logging
import collections
from typing import Deque, Dict, Tuple, List
from typing import List, Tuple
from functools import wraps
import logging
@StandardDecorator()
def long_term_memory_to_LRU_transfer(long_term_memory: LongTermMemory, LRU_memory: LRUMemory) -> None:
    """
    Transfers relevant information from long-term memory to LRU memory for efficient decision-making.

    Args:
        long_term_memory (LongTermMemory): The long-term memory instance.
        LRU_memory (LRUMemory): The LRU memory instance.
    """
    for key, value in long_term_memory.memory.items():
        board, move, score = value
        LRU_memory.store(board, move, score)
        del long_term_memory.memory[key]