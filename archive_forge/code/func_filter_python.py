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
def filter_python(version: t.Optional[str], versions: t.Optional[c.Sequence[str]]) -> t.Optional[str]:
    """If a Python version is given and is in the given version list, return that Python version, otherwise return None."""
    return version if version in versions else None