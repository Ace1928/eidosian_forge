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
def complete_remote_stage(prefix: str, **_) -> list[str]:
    """Return a list of supported stages matching the given prefix."""
    return [stage for stage in ('prod', 'dev') if stage.startswith(prefix)]