import dataclasses
import importlib
import json
import threading
import warnings
from collections import defaultdict, deque, namedtuple, OrderedDict
from typing import (
def arg_tree_leaves(*args: PyTree, **kwargs: PyTree) -> List[Any]:
    """Get a flat list of arguments to this function

    A slightly faster version of tree_leaves((args, kwargs))
    """
    leaves: List[Any] = []
    for a in args:
        _tree_leaves_helper(a, leaves)
    for a in kwargs.values():
        _tree_leaves_helper(a, leaves)
    return leaves