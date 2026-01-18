from __future__ import annotations
import collections.abc as c
import contextlib
import json
import os
import re
import shlex
import sys
import tempfile
import textwrap
import typing as t
from .constants import (
from .encoding import (
from .util import (
from .io import (
from .data import (
from .provider.layout import (
from .host_configs import (
def create_result_directories(args: CommonConfig) -> None:
    """Create result directories."""
    if args.explain:
        return
    make_dirs(ResultType.COVERAGE.path)
    make_dirs(ResultType.DATA.path)