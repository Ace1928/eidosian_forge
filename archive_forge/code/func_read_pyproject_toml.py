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
def read_pyproject_toml(ctx: click.Context, param: click.Parameter, value: Optional[str]) -> Optional[str]:
    """Inject Black configuration from "pyproject.toml" into defaults in `ctx`.

    Returns the path to a successfully found and read configuration file, None
    otherwise.
    """
    if not value:
        value = find_pyproject_toml(ctx.params.get('src', ()), ctx.params.get('stdin_filename', None))
        if value is None:
            return None
    try:
        config = parse_pyproject_toml(value)
    except (OSError, ValueError) as e:
        raise click.FileError(filename=value, hint=f'Error reading configuration file: {e}') from None
    if not config:
        return None
    else:
        spellcheck_pyproject_toml_keys(ctx, list(config), value)
        config = {k: str(v) if not isinstance(v, (list, dict)) else v for k, v in config.items()}
    target_version = config.get('target_version')
    if target_version is not None and (not isinstance(target_version, list)):
        raise click.BadOptionUsage('target-version', 'Config key target-version must be a list')
    exclude = config.get('exclude')
    if exclude is not None and (not isinstance(exclude, str)):
        raise click.BadOptionUsage('exclude', 'Config key exclude must be a string')
    extend_exclude = config.get('extend_exclude')
    if extend_exclude is not None and (not isinstance(extend_exclude, str)):
        raise click.BadOptionUsage('extend-exclude', 'Config key extend-exclude must be a string')
    line_ranges = config.get('line_ranges')
    if line_ranges is not None:
        raise click.BadOptionUsage('line-ranges', 'Cannot use line-ranges in the pyproject.toml file.')
    default_map: Dict[str, Any] = {}
    if ctx.default_map:
        default_map.update(ctx.default_map)
    default_map.update(config)
    ctx.default_map = default_map
    return value