import functools
import logging
import os
import platform
import subprocess
from typing import Dict, List, Optional, Union
from langsmith.utils import get_docker_compose_command
from langsmith.env._git import exec_git
@functools.lru_cache(maxsize=1)
def get_docker_version() -> Optional[str]:
    import subprocess
    try:
        docker_version = subprocess.check_output(['docker', '--version']).decode('utf-8').strip()
    except FileNotFoundError:
        docker_version = 'unknown'
    except:
        return None
    return docker_version