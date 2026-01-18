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
def get_coverage_files(language: str, path: t.Optional[str]=None) -> list[str]:
    """Return the list of coverage file paths for the given language."""
    coverage_dir = path or ResultType.COVERAGE.path
    try:
        coverage_files = [os.path.join(coverage_dir, f) for f in os.listdir(coverage_dir) if '=coverage.' in f and '=%s' % language in f]
    except FileNotFoundError:
        return []
    return coverage_files