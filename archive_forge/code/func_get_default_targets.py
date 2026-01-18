from __future__ import annotations
import abc
import dataclasses
import enum
import os
import pickle
import sys
import typing as t
from .constants import (
from .io import (
from .completion import (
from .util import (
def get_default_targets(self, context: HostContext) -> list[ControllerConfig]:
    """Return the default targets for this host config."""
    return [ControllerConfig(python=NativePythonConfig(version=version, path=path)) for version, path in get_available_python_versions().items()]