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
def controller_python(version: t.Optional[str]) -> t.Optional[str]:
    """If a Python version is given and is supported by the controller, return that Python version, otherwise return None."""
    return filter_python(version, CONTROLLER_PYTHON_VERSIONS)