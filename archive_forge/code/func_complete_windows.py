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
def complete_windows(prefix: str, parsed_args: argparse.Namespace, **_) -> list[str]:
    """Return a list of supported Windows versions matching the given prefix, excluding versions already parsed from the command line."""
    return [i for i in get_windows_version_choices() if i.startswith(prefix) and (not parsed_args.windows or i not in parsed_args.windows)]