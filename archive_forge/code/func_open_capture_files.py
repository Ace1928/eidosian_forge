import sys
import os
import argparse
import logging
import pathlib
import ray._private.ray_constants as ray_constants
from ray.core.generated import (
from ray._private.utils import open_log
from ray._private.ray_logging import (
from ray._private.utils import (
from ray._private.process_watcher import create_check_raylet_task
import_libs()
import runtime_env_consts  # noqa: E402
from runtime_env_agent import RuntimeEnvAgent  # noqa: E402
from aiohttp import web  # noqa: E402
def open_capture_files(log_dir):
    filename = 'runtime_env_agent'
    return (open_log(pathlib.Path(log_dir) / f'{filename}.out'), open_log(pathlib.Path(log_dir) / f'{filename}.err'))