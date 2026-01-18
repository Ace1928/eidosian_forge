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
def filter_profiles_for_target(args: IntegrationConfig, profiles: list[THostProfile], target: IntegrationTarget) -> list[THostProfile]:
    """Return a list of profiles after applying target filters."""
    if target.target_type == IntegrationTargetType.CONTROLLER:
        profile_filter = get_target_filter(args, [args.controller], True)
    elif target.target_type == IntegrationTargetType.TARGET:
        profile_filter = get_target_filter(args, args.targets, False)
    else:
        raise Exception(f'Unhandled test type for target "{target.name}": {target.target_type.name.lower()}')
    profiles = profile_filter.filter_profiles(profiles, target)
    return profiles