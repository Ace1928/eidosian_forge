import logging
import sys
from contextlib import contextmanager
from functools import wraps
from typing import Any, Dict, Mapping, Union
def merge_result(command, res):
    """
    Merge all items in `res` into a list.

    This command is used when sending a command to multiple nodes
    and the result from each node should be merged into a single list.

    res : 'dict'
    """
    result = set()
    for v in res.values():
        for value in v:
            result.add(value)
    return list(result)