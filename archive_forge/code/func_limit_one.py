from __future__ import annotations
import abc
import collections.abc as c
import contextlib
import dataclasses
import enum
import os
import re
import typing as t
@property
def limit_one(self) -> bool:
    """True if only one target is allowed, otherwise False."""
    return not self.use_list