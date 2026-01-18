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
@staticmethod
def requires_flush(path, path_used, was_multipath):
    """Check if a device needs to be flushed when detaching.

        A device representing a single path connection to a volume must only be
        flushed if it has been used directly by Nova or Cinder to write data.

        If the path has been used via a multipath DM or if the device was part
        of a multipath but a different single path was used for I/O (instead of
        the multipath) then we don't need to flush.
        """
    if not path_used:
        return False
    path = os.path.realpath(path)
    path_used = os.path.realpath(path_used)
    if path_used == path:
        return True
    return not was_multipath and '/dev' != os.path.split(path_used)[0]