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
def filename_in_pip_list(filename: str) -> bool:
    for pip_entry in pip_list:
        if filename in pip_entry:
            return True
    return False