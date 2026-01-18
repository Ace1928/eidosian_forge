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
def get_scsi_wwn(self, path: str) -> str:
    """Read the WWN from page 0x83 value for a SCSI device."""
    out, _err = self._execute('/lib/udev/scsi_id', '--page', '0x83', '--whitelisted', path, run_as_root=True, root_helper=self._root_helper)
    return out.strip()