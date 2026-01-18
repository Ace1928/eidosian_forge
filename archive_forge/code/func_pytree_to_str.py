import dataclasses
import importlib
import json
import threading
import warnings
from collections import defaultdict, deque, namedtuple, OrderedDict
from typing import (
def pytree_to_str(treespec: TreeSpec) -> str:
    warnings.warn('pytree_to_str is deprecated. Please use treespec_dumps')
    return treespec_dumps(treespec)