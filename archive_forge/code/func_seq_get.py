from __future__ import annotations
import datetime
import inspect
import logging
import re
import sys
import typing as t
from collections.abc import Collection, Set
from contextlib import contextmanager
from copy import copy
from enum import Enum
from itertools import count
def seq_get(seq: t.Sequence[T], index: int) -> t.Optional[T]:
    """Returns the value in `seq` at position `index`, or `None` if `index` is out of bounds."""
    try:
        return seq[index]
    except IndexError:
        return None