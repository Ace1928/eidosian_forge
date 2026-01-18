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
def run_setup_targets(args: IntegrationConfig, host_state: HostState, test_dir: str, target_names: c.Sequence[str], targets_dict: dict[str, IntegrationTarget], targets_executed: set[str], inventory_path: str, coverage_manager: CoverageManager, always: bool):
    """Run setup targets."""
    for target_name in target_names:
        if not always and target_name in targets_executed:
            continue
        target = targets_dict[target_name]
        if not args.explain:
            remove_tree(test_dir)
            make_dirs(test_dir)
        if target.script_path:
            command_integration_script(args, host_state, target, test_dir, inventory_path, coverage_manager)
        else:
            command_integration_role(args, host_state, target, None, test_dir, inventory_path, coverage_manager)
        targets_executed.add(target_name)