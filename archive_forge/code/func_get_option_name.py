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
def get_option_name(name: str) -> str:
    """Return a command-line option name from the given option name."""
    if name == 'targets':
        name = 'target'
    return f'--{name.replace('_', '-')}'