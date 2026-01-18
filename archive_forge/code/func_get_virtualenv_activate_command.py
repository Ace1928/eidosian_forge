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
@classmethod
def get_virtualenv_activate_command(cls, target_dir: str) -> List[str]:
    virtualenv_path = cls.get_virtualenv_path(target_dir)
    if _WIN32:
        cmd = [os.path.join(virtualenv_path, 'Scripts', 'activate.bat')]
    else:
        cmd = ['source', os.path.join(virtualenv_path, 'bin/activate')]
    return cmd + ['1>&2', '&&']