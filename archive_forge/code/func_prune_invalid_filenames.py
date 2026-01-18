from __future__ import annotations
import os
import typing as t
from .....encoding import (
from .....data import (
from .....util_common import (
from .....executor import (
from .....provisioning import (
from ... import (
from . import (
from . import (
def prune_invalid_filenames(args: CoverageAnalyzeTargetsGenerateConfig, results: dict[str, t.Any], collection_search_re: t.Optional[t.Pattern]=None) -> None:
    """Remove invalid filenames from the given result set."""
    path_checker = PathChecker(args, collection_search_re)
    for path in list(results.keys()):
        if not path_checker.check_path(path):
            del results[path]