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
def initialize_coverage(args: CoverageConfig, host_state: HostState) -> coverage_module:
    """Delegate execution if requested, install requirements, then import and return the coverage module. Raises an exception if coverage is not available."""
    configure_pypi_proxy(args, host_state.controller_profile)
    install_requirements(args, host_state.controller_profile.python, coverage=True)
    try:
        import coverage
    except ImportError:
        coverage = None
    coverage_required_version = CONTROLLER_COVERAGE_VERSION.coverage_version
    if not coverage:
        raise ApplicationError(f'Version {coverage_required_version} of the Python "coverage" module must be installed to use this command.')
    if coverage.__version__ != coverage_required_version:
        raise ApplicationError(f'Version {coverage_required_version} of the Python "coverage" module is required. Version {coverage.__version__} was found.')
    return coverage