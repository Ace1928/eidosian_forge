from __future__ import annotations
import collections.abc as c
import os
from . import (
from ...constants import (
from ...test import (
from ...target import (
from ...util import (
from ...util_common import (
from ...ansible_util import (
from ...python_requirements import (
from ...config import (
from ...coverage_util import (
from ...data import (
from ...host_configs import (
from ...venv import (
@cache
def get_ansible_test_python_path() -> str:
    """
    Return a directory usable for PYTHONPATH, containing only the ansible-test collection loader.
    The temporary directory created will be cached for the lifetime of the process and cleaned up at exit.
    """
    python_path = create_temp_dir(prefix='ansible-test-')
    return python_path