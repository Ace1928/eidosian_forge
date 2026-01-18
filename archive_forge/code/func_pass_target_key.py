from __future__ import annotations
import collections.abc as c
import re
import typing as t
from .....executor import (
from .....provisioning import (
from . import (
from . import (
def pass_target_key(value: TargetKey) -> TargetKey:
    """Return the given target key unmodified."""
    return value