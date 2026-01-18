from __future__ import annotations
from functools import lru_cache
import collections
import enum
import os
import re
import typing as T
def extend_preserving_lflags(self, iterable: T.Iterable[str]) -> None:
    normal_flags = []
    lflags = []
    for i in iterable:
        if i not in self.always_dedup_args and (i.startswith('-l') or i.startswith('-L')):
            lflags.append(i)
        else:
            normal_flags.append(i)
    self.extend(normal_flags)
    self.extend_direct(lflags)