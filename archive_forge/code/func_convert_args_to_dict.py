import os
import pathlib
import re
import sys
import time
import traceback
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple
import click
import watchfiles
import yaml
import ray
from ray import serve
from ray._private.utils import import_attr
from ray.autoscaler._private.cli_logger import cli_logger
from ray.dashboard.modules.dashboard_sdk import parse_runtime_env_args
from ray.dashboard.modules.serve.sdk import ServeSubmissionClient
from ray.serve._private import api as _private_api
from ray.serve._private.constants import (
from ray.serve._private.deployment_graph_build import build as pipeline_build
from ray.serve._private.deployment_graph_build import (
from ray.serve.config import DeploymentMode, ProxyLocation, gRPCOptions
from ray.serve.deployment import Application, deployment_to_schema
from ray.serve.schema import (
def convert_args_to_dict(args: Tuple[str]) -> Dict[str, str]:
    args_dict = dict()
    for arg in args:
        split = arg.split('=')
        if len(split) != 2:
            raise click.ClickException(f"Invalid application argument '{arg}', must be of the form '<key>=<val>'.")
        args_dict[split[0]] = split[1]
    return args_dict