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
def echo_scsi_command(self, path, content) -> None:
    """Used to echo strings to scsi subsystem."""
    args = ['-a', path]
    kwargs = dict(process_input=content, run_as_root=True, root_helper=self._root_helper)
    self._execute('tee', *args, **kwargs)