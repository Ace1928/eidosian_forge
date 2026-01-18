import functools
import logging
import os
import platform
import subprocess
from typing import Dict, List, Optional, Union
from langsmith.utils import get_docker_compose_command
from langsmith.env._git import exec_git
@functools.lru_cache(maxsize=1)
def get_runtime_environment() -> dict:
    """Get information about the environment."""
    from langsmith import __version__
    shas = get_release_shas()
    return {'sdk': 'langsmith-py', 'sdk_version': __version__, 'library': 'langsmith', 'platform': platform.platform(), 'runtime': 'python', 'py_implementation': platform.python_implementation(), 'runtime_version': platform.python_version(), 'langchain_version': get_langchain_environment(), **shas}