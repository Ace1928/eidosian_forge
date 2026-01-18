from __future__ import annotations
import os
import urllib.parse
from .io import (
from .config import (
from .host_configs import (
from .util import (
from .util_common import (
from .docker_util import (
from .containers import (
from .ansible_util import (
from .host_profiles import (
from .inventory import (
def cleanup_pypi_proxy() -> None:
    """Undo changes made to configure the PyPI proxy."""
    run_playbook(args, inventory_path, 'pypi_proxy_restore.yml', capture=True)