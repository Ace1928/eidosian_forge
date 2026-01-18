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
def get_windows_platform_choices() -> list[str]:
    """Return a list of supported Windows versions matching the given prefix."""
    return sorted((f'windows/{windows.version}' for windows in filter_completion(windows_completion()).values()))