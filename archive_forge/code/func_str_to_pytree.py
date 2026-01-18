import dataclasses
import importlib
import json
import threading
import warnings
from collections import defaultdict, deque, namedtuple, OrderedDict
from typing import (
def str_to_pytree(json: str) -> TreeSpec:
    warnings.warn('str_to_pytree is deprecated. Please use treespec_loads')
    return treespec_loads(json)