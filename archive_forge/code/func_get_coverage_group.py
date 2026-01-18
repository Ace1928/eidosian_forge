from __future__ import annotations
import collections.abc as c
import os
import json
import typing as t
from ...target import (
from ...io import (
from ...util import (
from ...util_common import (
from ...executor import (
from ...data import (
from ...host_configs import (
from ...provisioning import (
from . import (
def get_coverage_group(args: CoverageCombineConfig, coverage_file: str) -> t.Optional[str]:
    """Return the name of the coverage group for the specified coverage file, or None if no group was found."""
    parts = os.path.basename(coverage_file).split('=', 4)
    if len(parts) != 5 or not parts[4].startswith('coverage.'):
        return None
    names = dict(command=parts[0], target=parts[1], environment=parts[2], version=parts[3])
    export_names = dict(version=parts[3])
    group = ''
    for part in COVERAGE_GROUPS:
        if part in args.group_by:
            group += '=%s' % names[part]
        elif args.export:
            group += '=%s' % export_names.get(part, 'various')
    if args.export:
        group = group.lstrip('=')
    return group