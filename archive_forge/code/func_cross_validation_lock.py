from __future__ import annotations
import contextlib
import enum
import inspect
import os
import re
import sys
import types
import typing as t
from ast import literal_eval
from .utils.bunch import Bunch
from .utils.descriptions import add_article, class_of, describe, repr_type
from .utils.getargspec import getargspec
from .utils.importstring import import_item
from .utils.sentinel import Sentinel
from .utils.warnings import deprecated_method, should_warn, warn
from all trait attributes.
@property
@contextlib.contextmanager
def cross_validation_lock(self) -> t.Any:
    """
        A contextmanager for running a block with our cross validation lock set
        to True.

        At the end of the block, the lock's value is restored to its value
        prior to entering the block.
        """
    if self._cross_validation_lock:
        yield
        return
    else:
        try:
            self._cross_validation_lock = True
            yield
        finally:
            self._cross_validation_lock = False