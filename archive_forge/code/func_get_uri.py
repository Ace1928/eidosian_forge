import asyncio
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from typing import Dict, List, Optional, Tuple
from contextlib import asynccontextmanager
from asyncio import create_task, get_running_loop
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.packaging import Protocol, parse_uri
from ray._private.runtime_env.plugin import RuntimeEnvPlugin
from ray._private.runtime_env.utils import check_output_cmd
from ray._private.utils import get_directory_size_bytes, try_to_create_directory
import ray
def get_uri(runtime_env: Dict) -> Optional[str]:
    """Return `"pip://<hashed_dependencies>"`, or None if no GC required."""
    pip = runtime_env.get('pip')
    if pip is not None:
        if isinstance(pip, dict):
            uri = 'pip://' + _get_pip_hash(pip_dict=pip)
        elif isinstance(pip, list):
            uri = 'pip://' + _get_pip_hash(pip_dict=dict(packages=pip))
        else:
            raise TypeError(f'pip field received by RuntimeEnvAgent must be list or dict, not {type(pip).__name__}.')
    else:
        uri = None
    return uri