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
@click.command()
@click.argument('cluster_config_file', required=True, type=str)
@click.option('--cluster-name', '-n', required=False, type=str, help='Override the configured cluster name.')
@click.option('--port', '-p', required=False, type=int, default=ray_constants.DEFAULT_DASHBOARD_PORT, help='The local port to forward to the dashboard')
@click.option('--remote-port', required=False, type=int, default=ray_constants.DEFAULT_DASHBOARD_PORT, help='The remote port your dashboard runs on')
@click.option('--no-config-cache', is_flag=True, default=False, help='Disable the local cluster config cache.')
@PublicAPI
def dashboard(cluster_config_file, cluster_name, port, remote_port, no_config_cache):
    """Port-forward a Ray cluster's dashboard to the local machine."""
    try:
        port_forward = [(port, remote_port)]
        click.echo('Attempting to establish dashboard locally at http://localhost:{}/ connected to remote port {}'.format(port, remote_port))
        exec_cluster(cluster_config_file, override_cluster_name=cluster_name, port_forward=port_forward, no_config_cache=no_config_cache)
        click.echo('Successfully established connection.')
    except Exception as e:
        raise click.ClickException('Failed to forward dashboard from remote port {1} to local port {0}. There are a couple possibilities: \n 1. The remote port is incorrectly specified \n 2. The local port {0} is already in use.\n The exception is: {2}'.format(port, remote_port, e)) from None