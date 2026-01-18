from __future__ import annotations
import argparse
import enum
import functools
import typing as t
from ..constants import (
from ..util import (
from ..completion import (
from ..cli.argparsing import (
from ..cli.argparsing.actions import (
from ..cli.actions import (
from ..cli.compat import (
from ..config import (
from .completers import (
from .converters import (
from .epilog import (
from ..ci import (
def add_legacy_environment_options(parser: argparse.ArgumentParser, controller_mode: ControllerMode, target_mode: TargetMode):
    """Add legacy options for controlling the test environment."""
    environment: argparse.ArgumentParser = parser.add_argument_group(title='environment arguments (mutually exclusive with "composite environment arguments" below)')
    add_environments_python(environment, target_mode)
    add_environments_host(environment, controller_mode, target_mode)