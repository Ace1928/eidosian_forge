from __future__ import annotations
import collections
import collections.abc as c
import enum
import os
import re
import itertools
import abc
import typing as t
from .encoding import (
from .io import (
from .util import (
from .data import (
def find_target_completion(target_func: c.Callable[[], c.Iterable[CompletionTarget]], prefix: str, short: bool) -> list[str]:
    """Return a list of targets from the given target function which match the given prefix."""
    try:
        targets = target_func()
        matches = list(walk_completion_targets(targets, prefix, short))
        return matches
    except Exception as ex:
        return ['%s' % ex]