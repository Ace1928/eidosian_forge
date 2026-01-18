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
def complete_network_platform_connection(prefix: str, parsed_args: argparse.Namespace, **_) -> list[str]:
    """Return a list of supported network platforms matching the given prefix, excluding connection platforms already parsed from the command line."""
    left = prefix.split('=')[0]
    images = sorted(set((image.platform for image in filter_completion(network_completion()).values())))
    return [i + '=' for i in images if i.startswith(left) and (not parsed_args.platform_connection or i not in [x[0] for x in parsed_args.platform_connection])]