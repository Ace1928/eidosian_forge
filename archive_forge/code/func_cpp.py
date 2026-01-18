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
@cli.command()
@click.option('--show-library-path', '-show', required=False, is_flag=True, help='Show the cpp include path and library path, if provided.')
@click.option('--generate-bazel-project-template-to', '-gen', required=False, type=str, help='The directory to generate the bazel project template to, if provided.')
@add_click_logging_options
def cpp(show_library_path, generate_bazel_project_template_to):
    """Show the cpp library path and generate the bazel project template."""
    if sys.platform == 'win32':
        cli_logger.error('Ray C++ API is not supported on Windows currently.')
        sys.exit(1)
    if not show_library_path and (not generate_bazel_project_template_to):
        raise ValueError("Please input at least one option of '--show-library-path' and '--generate-bazel-project-template-to'.")
    raydir = os.path.abspath(os.path.dirname(ray.__file__))
    cpp_dir = os.path.join(raydir, 'cpp')
    cpp_templete_dir = os.path.join(cpp_dir, 'example')
    include_dir = os.path.join(cpp_dir, 'include')
    lib_dir = os.path.join(cpp_dir, 'lib')
    if not os.path.isdir(cpp_dir):
        raise ValueError('Please install ray with C++ API by "pip install ray[cpp]".')
    if show_library_path:
        cli_logger.print('Ray C++ include path {} ', cf.bold(f'{include_dir}'))
        cli_logger.print('Ray C++ library path {} ', cf.bold(f'{lib_dir}'))
    if generate_bazel_project_template_to:
        if os.path.exists(generate_bazel_project_template_to):
            shutil.rmtree(generate_bazel_project_template_to)
        shutil.copytree(cpp_templete_dir, generate_bazel_project_template_to)
        out_include_dir = os.path.join(generate_bazel_project_template_to, 'thirdparty/include')
        if os.path.exists(out_include_dir):
            shutil.rmtree(out_include_dir)
        shutil.copytree(include_dir, out_include_dir)
        out_lib_dir = os.path.join(generate_bazel_project_template_to, 'thirdparty/lib')
        if os.path.exists(out_lib_dir):
            shutil.rmtree(out_lib_dir)
        shutil.copytree(lib_dir, out_lib_dir)
        cli_logger.print('Project template generated to {}', cf.bold(f'{os.path.abspath(generate_bazel_project_template_to)}'))
        cli_logger.print('To build and run this template, run')
        cli_logger.print(cf.bold(f'    cd {os.path.abspath(generate_bazel_project_template_to)} && bash run.sh'))