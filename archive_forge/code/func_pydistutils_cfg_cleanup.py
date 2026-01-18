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
def pydistutils_cfg_cleanup() -> None:
    """Remove custom PyPI config."""
    display.info('Removing custom PyPI config: %s' % pydistutils_cfg_path, verbosity=1)
    os.remove(pydistutils_cfg_path)