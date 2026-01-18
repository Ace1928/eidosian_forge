from __future__ import annotations
import dataclasses
import os
import typing as t
from .io import (
from .util import (
from .util_common import (
from .core_ci import (
@property
def bootstrap_type(self) -> str:
    """The bootstrap type to pass to the bootstrapping script."""
    return self.__class__.__name__.replace('Bootstrap', '').lower()