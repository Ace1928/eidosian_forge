from __future__ import annotations
import glob
import os
import re
import time
from typing import Optional
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_utils import excutils
from os_brick import constants
from os_brick import exception
from os_brick import executor
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick import utils
def get_name_from_path(self, path) -> Optional[str]:
    """Translates /dev/disk/by-path/ entry to /dev/sdX."""
    name = os.path.realpath(path)
    if name.startswith('/dev/'):
        return name
    else:
        return None