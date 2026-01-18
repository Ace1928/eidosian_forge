import contextlib
import enum
import functools
import logging
import os
import pathlib
import subprocess
import threading
from typing import (
import requests
from urllib3.util import Retry
from langsmith import schemas as ls_schemas
@functools.lru_cache(maxsize=1)
def get_docker_compose_command() -> List[str]:
    """Get the correct docker compose command for this system."""
    try:
        subprocess.check_call(['docker', 'compose', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return ['docker', 'compose']
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            subprocess.check_call(['docker-compose', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return ['docker-compose']
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise ValueError("Neither 'docker compose' nor 'docker-compose' commands are available. Please install the Docker server following the instructions for your operating system at https://docs.docker.com/engine/install/")