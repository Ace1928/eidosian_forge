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
def find_new_name(taken: t.Collection[str], base: str) -> str:
    """
    Searches for a new name.

    Args:
        taken: A collection of taken names.
        base: Base name to alter.

    Returns:
        The new, available name.
    """
    if base not in taken:
        return base
    i = 2
    new = f'{base}_{i}'
    while new in taken:
        i += 1
        new = f'{base}_{i}'
    return new