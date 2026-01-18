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
def get_future_imports(node: Node) -> Set[str]:
    """Return a set of __future__ imports in the file."""
    imports: Set[str] = set()

    def get_imports_from_children(children: List[LN]) -> Generator[str, None, None]:
        for child in children:
            if isinstance(child, Leaf):
                if child.type == token.NAME:
                    yield child.value
            elif child.type == syms.import_as_name:
                orig_name = child.children[0]
                assert isinstance(orig_name, Leaf), 'Invalid syntax parsing imports'
                assert orig_name.type == token.NAME, 'Invalid syntax parsing imports'
                yield orig_name.value
            elif child.type == syms.import_as_names:
                yield from get_imports_from_children(child.children)
            else:
                raise AssertionError('Invalid syntax parsing imports')
    for child in node.children:
        if child.type != syms.simple_stmt:
            break
        first_child = child.children[0]
        if isinstance(first_child, Leaf):
            if len(child.children) == 2 and first_child.type == token.STRING and (child.children[1].type == token.NEWLINE):
                continue
            break
        elif first_child.type == syms.import_from:
            module_name = first_child.children[1]
            if not isinstance(module_name, Leaf) or module_name.value != '__future__':
                break
            imports |= set(get_imports_from_children(first_child.children[3:]))
        else:
            break
    return imports