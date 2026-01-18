from __future__ import annotations
import abc
import os
import shutil
import tempfile
import typing as t
import zipfile
from ...io import (
from ...ansible_util import (
from ...config import (
from ...util import (
from ...util_common import (
from ...coverage_util import (
from ...host_configs import (
from ...data import (
from ...host_profiles import (
from ...provisioning import (
from ...connections import (
from ...inventory import (
def get_handler_type(config_type: t.Type[HostConfig]) -> t.Optional[t.Type[CoverageHandler]]:
    """Return the coverage handler type associated with the given host config type if found, otherwise return None."""
    queue = [config_type]
    type_map = get_config_handler_type_map()
    while queue:
        config_type = queue.pop(0)
        handler_type = type_map.get(config_type)
        if handler_type:
            return handler_type
        queue.extend(config_type.__bases__)
    return None