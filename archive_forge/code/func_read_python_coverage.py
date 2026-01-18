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
def read_python_coverage(path: str, coverage: coverage_module) -> PythonArcs:
    """Return coverage arcs from the specified coverage file. Raises a CoverageError exception if coverage cannot be read."""
    try:
        return read_python_coverage_native(path, coverage)
    except CoverageError as ex:
        schema_version = get_coverage_file_schema_version(path)
        if schema_version == CONTROLLER_COVERAGE_VERSION.schema_version:
            raise CoverageError(path, f'Unexpected failure reading supported schema version {schema_version}.') from ex
    if schema_version == 0:
        return read_python_coverage_legacy(path)
    raise CoverageError(path, f'Unsupported schema version: {schema_version}')