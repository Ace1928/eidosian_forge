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
def get_files_needed(target_dependencies: list[IntegrationTarget]) -> list[str]:
    """Return a list of files needed by the given list of target dependencies."""
    files_needed: list[str] = []
    for target_dependency in target_dependencies:
        files_needed += target_dependency.needs_file
    files_needed = sorted(set(files_needed))
    invalid_paths = [path for path in files_needed if not os.path.isfile(path)]
    if invalid_paths:
        raise ApplicationError('Invalid "needs/file/*" aliases:\n%s' % '\n'.join(invalid_paths))
    return files_needed