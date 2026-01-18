import copy
import json
import logging
import os
import signal
import subprocess
import sys
import time
import traceback
import urllib
import urllib.parse
import warnings
import shutil
from datetime import datetime
from typing import Optional, Set, List, Tuple
import click
import psutil
import yaml
import ray
import ray._private.ray_constants as ray_constants
import ray._private.services as services
from ray._private.utils import (
from ray._private.internal_api import memory_summary
from ray._private.storage import _load_class
from ray._private.usage import usage_lib
from ray.autoscaler._private.cli_logger import add_click_logging_options, cf, cli_logger
from ray.autoscaler._private.commands import (
from ray.autoscaler._private.constants import RAY_PROCESSES
from ray.autoscaler._private.fake_multi_node.node_provider import FAKE_HEAD_NODE_ID
from ray.util.annotations import PublicAPI
@cli.command(name='health-check', hidden=True)
@click.option('--address', required=False, type=str, help='Override the address to connect to.')
@click.option('--redis_password', required=False, type=str, default=ray_constants.REDIS_DEFAULT_PASSWORD, help='Connect to ray with redis_password.')
@click.option('--component', required=False, type=str, help='Health check for a specific component. Currently supports: [ray_client_server]')
@click.option('--skip-version-check', is_flag=True, default=False, help='Skip comparison of GCS version with local Ray version.')
def healthcheck(address, redis_password, component, skip_version_check):
    """
    This is NOT a public api.

    Health check a Ray or a specific component. Exit code 0 is healthy.
    """
    address = services.canonicalize_bootstrap_address_or_die(address)
    if not component:
        try:
            if ray._raylet.check_health(address, skip_version_check=skip_version_check):
                sys.exit(0)
        except Exception:
            traceback.print_exc()
            pass
        sys.exit(1)
    gcs_client = ray._raylet.GcsClient(address=address)
    ray.experimental.internal_kv._initialize_internal_kv(gcs_client)
    report_str = ray.experimental.internal_kv._internal_kv_get(component, namespace=ray_constants.KV_NAMESPACE_HEALTHCHECK)
    if not report_str:
        sys.exit(1)
    report = json.loads(report_str)
    cur_time = time.time()
    report_time = float(report['time'])
    delta = cur_time - report_time
    time_ok = delta < ray._private.ray_constants.HEALTHCHECK_EXPIRATION_S
    if time_ok:
        sys.exit(0)
    else:
        sys.exit(1)