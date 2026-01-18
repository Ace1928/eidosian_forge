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
@property
def dependency_map(self) -> dict[str, set[IntegrationTarget]]:
    """The dependency map of integration test targets."""
    return self.get('dependency_map', lambda: generate_dependency_map(self.integration_targets))