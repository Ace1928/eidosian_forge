from __future__ import annotations
import dataclasses
import enum
import os
import sys
import typing as t
from .util import (
from .util_common import (
from .metadata import (
from .data import (
from .host_configs import (
def only_targets(self, target_type: t.Type[THostConfig]) -> list[THostConfig]:
    """
        Return a list of target host configurations.
        Requires that there are one or more targets, all the specified type.
        """
    if not self.targets:
        raise Exception('There must be one or more targets.')
    assert type_guard(self.targets, target_type)
    return t.cast(list[THostConfig], self.targets)