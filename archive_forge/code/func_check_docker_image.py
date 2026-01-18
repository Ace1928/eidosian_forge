from pathlib import Path
from typing import Any, Dict
from ray.autoscaler._private.cli_logger import cli_logger
def check_docker_image(cname, docker_cmd):
    return _check_helper(cname, '.Config.Image', docker_cmd)