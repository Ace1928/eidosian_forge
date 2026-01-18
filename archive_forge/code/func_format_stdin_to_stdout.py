import io
import json
import platform
import re
import sys
import tokenize
import traceback
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from enum import Enum
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import (
import click
from click.core import ParameterSource
from mypy_extensions import mypyc_attr
from pathspec import PathSpec
from pathspec.patterns.gitwildmatch import GitWildMatchPatternError
from _black_version import version as __version__
from black.cache import Cache
from black.comments import normalize_fmt_off
from black.const import (
from black.files import (
from black.handle_ipynb_magics import (
from black.linegen import LN, LineGenerator, transform_line
from black.lines import EmptyLineTracker, LinesBlock
from black.mode import FUTURE_FLAG_TO_FEATURE, VERSION_TO_FEATURES, Feature
from black.mode import Mode as Mode  # re-exported
from black.mode import Preview, TargetVersion, supports_feature
from black.nodes import (
from black.output import color_diff, diff, dump_to_file, err, ipynb_diff, out
from black.parsing import (  # noqa F401
from black.ranges import (
from black.report import Changed, NothingChanged, Report
from black.trans import iter_fexpr_spans
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def format_stdin_to_stdout(fast: bool, *, content: Optional[str]=None, write_back: WriteBack=WriteBack.NO, mode: Mode, lines: Collection[Tuple[int, int]]=()) -> bool:
    """Format file on stdin. Return True if changed.

    If content is None, it's read from sys.stdin.

    If `write_back` is YES, write reformatted code back to stdout. If it is DIFF,
    write a diff to stdout. The `mode` argument is passed to
    :func:`format_file_contents`.
    """
    then = datetime.now(timezone.utc)
    if content is None:
        src, encoding, newline = decode_bytes(sys.stdin.buffer.read())
    else:
        src, encoding, newline = (content, 'utf-8', '')
    dst = src
    try:
        dst = format_file_contents(src, fast=fast, mode=mode, lines=lines)
        return True
    except NothingChanged:
        return False
    finally:
        f = io.TextIOWrapper(sys.stdout.buffer, encoding=encoding, newline=newline, write_through=True)
        if write_back == WriteBack.YES:
            if dst and dst[-1] != '\n':
                dst += '\n'
            f.write(dst)
        elif write_back in (WriteBack.DIFF, WriteBack.COLOR_DIFF):
            now = datetime.now(timezone.utc)
            src_name = f'STDIN\t{then}'
            dst_name = f'STDOUT\t{now}'
            d = diff(src, dst, src_name, dst_name)
            if write_back == WriteBack.COLOR_DIFF:
                d = color_diff(d)
                f = wrap_stream_for_windows(f)
            f.write(d)
        f.detach()