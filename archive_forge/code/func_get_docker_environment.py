import functools
import logging
import os
import platform
import subprocess
from typing import Dict, List, Optional, Union
from langsmith.utils import get_docker_compose_command
from langsmith.env._git import exec_git
@functools.lru_cache(maxsize=1)
def get_docker_environment() -> dict:
    """Get information about the environment."""
    compose_command = _get_compose_command()
    return {'docker_version': get_docker_version(), 'docker_compose_command': ' '.join(compose_command) if compose_command is not None else None, 'docker_compose_version': get_docker_compose_version()}