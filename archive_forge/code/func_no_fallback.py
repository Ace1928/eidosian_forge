from __future__ import annotations
import argparse
import collections.abc as c
import dataclasses
import enum
import os
import types
import typing as t
from ..constants import (
from ..util import (
from ..docker_util import (
from ..completion import (
from ..host_configs import (
from ..data import (
@property
def no_fallback(self) -> bool:
    """Return True if no fallback is acceptable for the controller (due to options not applying to the target), otherwise return False."""
    return self in (TargetMode.WINDOWS_INTEGRATION, TargetMode.NETWORK_INTEGRATION, TargetMode.NO_TARGETS)