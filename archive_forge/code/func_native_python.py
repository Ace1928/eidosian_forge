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
def native_python(options: LegacyHostOptions) -> t.Optional[NativePythonConfig]:
    """Return a NativePythonConfig for the given version if it is not None, otherwise return None."""
    if not options.python and (not options.python_interpreter):
        return None
    return NativePythonConfig(version=options.python, path=options.python_interpreter)