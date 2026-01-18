import bisect
import dataclasses
import dis
import sys
from typing import Any, Set, Union
def get_indexof(insts):
    """
    Get a mapping from instruction memory address to index in instruction list.
    Additionally checks that each instruction only appears once in the list.
    """
    indexof = {}
    for i, inst in enumerate(insts):
        assert inst not in indexof
        indexof[inst] = i
    return indexof