from __future__ import annotations
from collections import ChainMap
import datetime
import inspect
from io import StringIO
import itertools
import pprint
import struct
import sys
from typing import TypeVar
import numpy as np
from pandas._libs.tslibs import Timestamp
from pandas.errors import UndefinedVariableError
def ensure_scope(level: int, global_dict=None, local_dict=None, resolvers=(), target=None) -> Scope:
    """Ensure that we are grabbing the correct scope."""
    return Scope(level + 1, global_dict=global_dict, local_dict=local_dict, resolvers=resolvers, target=target)