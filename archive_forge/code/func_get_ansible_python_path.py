from __future__ import annotations
import json
import os
import shutil
import typing as t
from .constants import (
from .io import (
from .util import (
from .util_common import (
from .config import (
from .data import (
from .python_requirements import (
from .host_configs import (
from .thread import (
@mutex
def get_ansible_python_path(args: CommonConfig) -> str:
    """
    Return a directory usable for PYTHONPATH, containing only the ansible package.
    If a temporary directory is required, it will be cached for the lifetime of the process and cleaned up at exit.
    """
    try:
        return get_ansible_python_path.python_path
    except AttributeError:
        pass
    if ANSIBLE_SOURCE_ROOT:
        python_path = os.path.dirname(ANSIBLE_LIB_ROOT)
    else:
        python_path = create_temp_dir(prefix='ansible-test-')
        os.symlink(ANSIBLE_LIB_ROOT, os.path.join(python_path, 'ansible'))
    if not args.explain:
        generate_egg_info(python_path)
    get_ansible_python_path.python_path = python_path
    return python_path