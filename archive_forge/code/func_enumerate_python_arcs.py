from __future__ import annotations
import collections.abc as c
import json
import os
import re
import typing as t
from ...encoding import (
from ...io import (
from ...util import (
from ...util_common import (
from ...config import (
from ...python_requirements import (
from ...target import (
from ...data import (
from ...pypi_proxy import (
from ...provisioning import (
from ...coverage_util import (
def enumerate_python_arcs(path: str, coverage: coverage_module, modules: dict[str, str], collection_search_re: t.Optional[t.Pattern], collection_sub_re: t.Optional[t.Pattern]) -> c.Generator[tuple[str, set[tuple[int, int]]], None, None]:
    """Enumerate Python code coverage arcs in the given file."""
    if os.path.getsize(path) == 0:
        display.warning('Empty coverage file: %s' % path, verbosity=2)
        return
    try:
        arc_data = read_python_coverage(path, coverage)
    except CoverageError as ex:
        display.error(str(ex))
        return
    for filename, arcs in arc_data.items():
        if not arcs:
            display.warning('No arcs found for "%s" in coverage file: %s' % (filename, path))
            continue
        filename = sanitize_filename(filename, modules=modules, collection_search_re=collection_search_re, collection_sub_re=collection_sub_re)
        if not filename:
            continue
        yield (filename, set(arcs))