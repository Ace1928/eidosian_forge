from __future__ import annotations
import dataclasses
import os
import sqlite3
import tempfile
import typing as t
from .config import (
from .io import (
from .util import (
from .data import (
from .util_common import (
from .host_configs import (
from .constants import (
from .thread import (
def generate_ansible_coverage_config() -> str:
    """Generate code coverage configuration for Ansible tests."""
    coverage_config = '\n[run]\nbranch = True\nconcurrency =\n    multiprocessing\n    thread\nparallel = True\n\nomit =\n    */python*/dist-packages/*\n    */python*/site-packages/*\n    */python*/distutils/*\n    */pyshared/*\n    */pytest\n    */AnsiballZ_*.py\n    */test/results/*\n'
    return coverage_config