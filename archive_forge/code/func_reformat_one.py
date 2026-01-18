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
@mypyc_attr(patchable=True)
def reformat_one(src: Path, fast: bool, write_back: WriteBack, mode: Mode, report: 'Report', *, lines: Collection[Tuple[int, int]]=()) -> None:
    """Reformat a single file under `src` without spawning child processes.

    `fast`, `write_back`, and `mode` options are passed to
    :func:`format_file_in_place` or :func:`format_stdin_to_stdout`.
    """
    try:
        changed = Changed.NO
        if str(src) == '-':
            is_stdin = True
        elif str(src).startswith(STDIN_PLACEHOLDER):
            is_stdin = True
            src = Path(str(src)[len(STDIN_PLACEHOLDER):])
        else:
            is_stdin = False
        if is_stdin:
            if src.suffix == '.pyi':
                mode = replace(mode, is_pyi=True)
            elif src.suffix == '.ipynb':
                mode = replace(mode, is_ipynb=True)
            if format_stdin_to_stdout(fast=fast, write_back=write_back, mode=mode, lines=lines):
                changed = Changed.YES
        else:
            cache = Cache.read(mode)
            if write_back not in (WriteBack.DIFF, WriteBack.COLOR_DIFF):
                if not cache.is_changed(src):
                    changed = Changed.CACHED
            if changed is not Changed.CACHED and format_file_in_place(src, fast=fast, write_back=write_back, mode=mode, lines=lines):
                changed = Changed.YES
            if write_back is WriteBack.YES and changed is not Changed.CACHED or (write_back is WriteBack.CHECK and changed is Changed.NO):
                cache.write([src])
        report.done(src, changed)
    except Exception as exc:
        if report.verbose:
            traceback.print_exc()
        report.failed(src, str(exc))