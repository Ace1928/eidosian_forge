import asyncio
import errno
import ipaddress
import logging
import os
import pathlib
import sys
import time
from math import floor
from packaging.version import Version
import ray.dashboard.optional_utils as dashboard_optional_utils
import ray.dashboard.utils as dashboard_utils
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
from ray._private.utils import get_or_create_event_loop
from ray._raylet import GcsClient
from ray.dashboard.dashboard_metrics import DashboardPrometheusMetrics
from ray.dashboard.optional_deps import aiohttp, hdrs
def setup_static_dir():
    build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'client', 'build')
    module_name = os.path.basename(os.path.dirname(__file__))
    if not os.path.isdir(build_dir):
        raise dashboard_utils.FrontendNotFoundError(errno.ENOENT, f'Dashboard build directory not found. If installing from source, please follow the additional steps required to build the dashboard(cd python/ray/{module_name}/client && npm ci && npm run build)', build_dir)
    static_dir = os.path.join(build_dir, 'static')
    routes.static('/static', static_dir, follow_symlinks=True)
    return build_dir