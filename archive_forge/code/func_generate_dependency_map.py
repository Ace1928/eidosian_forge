from __future__ import annotations
import collections.abc as c
import contextlib
import datetime
import json
import os
import re
import shutil
import tempfile
import time
import typing as t
from ...encoding import (
from ...ansible_util import (
from ...executor import (
from ...python_requirements import (
from ...ci import (
from ...target import (
from ...config import (
from ...io import (
from ...util import (
from ...util_common import (
from ...coverage_util import (
from ...cache import (
from .cloud import (
from ...data import (
from ...host_configs import (
from ...host_profiles import (
from ...provisioning import (
from ...pypi_proxy import (
from ...inventory import (
from .filters import (
from .coverage import (
def generate_dependency_map(integration_targets: list[IntegrationTarget]) -> dict[str, set[IntegrationTarget]]:
    """Analyze the given list of integration test targets and return a dictionary expressing target names and the targets on which they depend."""
    targets_dict = dict(((target.name, target) for target in integration_targets))
    target_dependencies = analyze_integration_target_dependencies(integration_targets)
    dependency_map: dict[str, set[IntegrationTarget]] = {}
    invalid_targets = set()
    for dependency, dependents in target_dependencies.items():
        dependency_target = targets_dict.get(dependency)
        if not dependency_target:
            invalid_targets.add(dependency)
            continue
        for dependent in dependents:
            if dependent not in dependency_map:
                dependency_map[dependent] = set()
            dependency_map[dependent].add(dependency_target)
    if invalid_targets:
        raise ApplicationError('Non-existent target dependencies: %s' % ', '.join(sorted(invalid_targets)))
    return dependency_map