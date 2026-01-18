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
def create_inventory(args: IntegrationConfig, host_state: HostState, inventory_path: str, target: IntegrationTarget) -> None:
    """Create inventory."""
    if isinstance(args, PosixIntegrationConfig):
        if target.target_type == IntegrationTargetType.CONTROLLER:
            display.info('Configuring controller inventory.', verbosity=1)
            create_controller_inventory(args, inventory_path, host_state.controller_profile)
        elif target.target_type == IntegrationTargetType.TARGET:
            display.info('Configuring target inventory.', verbosity=1)
            create_posix_inventory(args, inventory_path, host_state.target_profiles, 'needs/ssh/' in target.aliases)
        else:
            raise Exception(f'Unhandled test type for target "{target.name}": {target.target_type.name.lower()}')
    elif isinstance(args, WindowsIntegrationConfig):
        display.info('Configuring target inventory.', verbosity=1)
        target_profiles = filter_profiles_for_target(args, host_state.target_profiles, target)
        create_windows_inventory(args, inventory_path, target_profiles)
    elif isinstance(args, NetworkIntegrationConfig):
        display.info('Configuring target inventory.', verbosity=1)
        target_profiles = filter_profiles_for_target(args, host_state.target_profiles, target)
        create_network_inventory(args, inventory_path, target_profiles)