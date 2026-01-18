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
def get_python_modules() -> dict[str, str]:
    """Return a dictionary of Ansible module names and their paths."""
    return dict(((target.module, target.path) for target in list(walk_module_targets()) if target.path.endswith('.py')))