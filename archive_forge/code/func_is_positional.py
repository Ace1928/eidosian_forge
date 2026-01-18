from __future__ import annotations
import collections
import collections.abc
import dataclasses
import enum
import functools
import inspect
import itertools
import numbers
import os
import sys
import typing
import warnings
from typing import (
import docstring_parser
import typing_extensions
from typing_extensions import (
from . import (
from ._typing import TypeForm
from .conf import _confstruct, _markers
def is_positional(self) -> bool:
    """Returns True if the argument should be positional in the commandline."""
    return _markers.Positional in self.markers or self.intern_name == _strings.dummy_field_name or (_markers.PositionalRequiredArgs in self.markers and self.default in MISSING_SINGLETONS)