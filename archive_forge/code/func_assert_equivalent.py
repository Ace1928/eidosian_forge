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
def assert_equivalent(src: str, dst: str) -> None:
    """Raise AssertionError if `src` and `dst` aren't equivalent."""
    try:
        src_ast = parse_ast(src)
    except Exception as exc:
        raise ASTSafetyError(f'cannot use --safe with this file; failed to parse source file AST: {exc}\nThis could be caused by running Black with an older Python version that does not support new syntax used in your source file.') from exc
    try:
        dst_ast = parse_ast(dst)
    except Exception as exc:
        log = dump_to_file(''.join(traceback.format_tb(exc.__traceback__)), dst)
        raise ASTSafetyError(f'INTERNAL ERROR: Black produced invalid code: {exc}. Please report a bug on https://github.com/psf/black/issues.  This invalid output might be helpful: {log}') from None
    src_ast_str = '\n'.join(stringify_ast(src_ast))
    dst_ast_str = '\n'.join(stringify_ast(dst_ast))
    if src_ast_str != dst_ast_str:
        log = dump_to_file(diff(src_ast_str, dst_ast_str, 'src', 'dst'))
        raise ASTSafetyError(f'INTERNAL ERROR: Black produced code that is not equivalent to the source.  Please report a bug on https://github.com/psf/black/issues.  This diff might be helpful: {log}') from None