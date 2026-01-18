import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.packaging import (
from ray._private.runtime_env.plugin import RuntimeEnvPlugin
from ray._private.utils import get_directory_size_bytes, try_to_create_directory
from ray.exceptions import RuntimeEnvSetupError
def set_pythonpath_in_context(python_path: str, context: RuntimeEnvContext):
    """Insert the path as the first entry in PYTHONPATH in the runtime env.

    This is compatible with users providing their own PYTHONPATH in env_vars,
    and is also compatible with the existing PYTHONPATH in the cluster.

    The import priority is as follows:
    this python_path arg > env_vars PYTHONPATH > existing cluster env PYTHONPATH.
    """
    if 'PYTHONPATH' in context.env_vars:
        python_path += os.pathsep + context.env_vars['PYTHONPATH']
    if 'PYTHONPATH' in os.environ:
        python_path += os.pathsep + os.environ['PYTHONPATH']
    context.env_vars['PYTHONPATH'] = python_path