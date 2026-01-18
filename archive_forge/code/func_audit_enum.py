from __future__ import annotations
import abc
import dataclasses
import enum
import os
import typing as t
from .constants import (
from .util import (
from .data import (
from .become import (
@property
def audit_enum(self) -> AuditMode:
    """The audit requirements for the container. Raises an exception if the value is invalid."""
    try:
        return AuditMode(self.audit)
    except ValueError:
        raise ValueError(f'Docker completion entry "{self.name}" has an invalid value "{self.audit}" for the "audit" setting.') from None