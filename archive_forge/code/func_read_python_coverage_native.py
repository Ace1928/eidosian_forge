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
def read_python_coverage_native(path: str, coverage: coverage_module) -> PythonArcs:
    """Return coverage arcs from the specified coverage file using the coverage API."""
    try:
        data = coverage.CoverageData(path)
        data.read()
        arcs = {filename: data.arcs(filename) for filename in data.measured_files()}
    except Exception as ex:
        raise CoverageError(path, f'Error reading coverage file using coverage API: {ex}') from ex
    return arcs