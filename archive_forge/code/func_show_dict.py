from __future__ import annotations
import datetime
import os
import platform
import sys
import typing as t
from ...config import (
from ...io import (
from ...util import (
from ...util_common import (
from ...docker_util import (
from ...constants import (
from ...ci import (
from ...timeout import (
def show_dict(data: dict[str, t.Any], verbose: dict[str, int], root_verbosity: int=0, path: t.Optional[list[str]]=None) -> None:
    """Show a dict with varying levels of verbosity."""
    path = path if path else []
    for key, value in sorted(data.items()):
        indent = '  ' * len(path)
        key_path = path + [key]
        key_name = '.'.join(key_path)
        verbosity = verbose.get(key_name, root_verbosity)
        if isinstance(value, (tuple, list)):
            display.info(indent + '%s:' % key, verbosity=verbosity)
            for item in value:
                display.info(indent + '  - %s' % item, verbosity=verbosity)
        elif isinstance(value, dict):
            min_verbosity = min([verbosity] + [v for k, v in verbose.items() if k.startswith('%s.' % key)])
            display.info(indent + '%s:' % key, verbosity=min_verbosity)
            show_dict(value, verbose, verbosity, key_path)
        else:
            display.info(indent + '%s: %s' % (key, value), verbosity=verbosity)