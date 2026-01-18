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
def is_multipath_running(enforce_multipath, root_helper, execute=None) -> bool:
    try:
        if execute is None:
            execute = priv_rootwrap.execute
        cmd = ('multipathd', 'show', 'status')
        out, _err = execute(*cmd, run_as_root=True, root_helper=root_helper)
        if out and out.startswith('error receiving packet'):
            raise putils.ProcessExecutionError('', out, 1, cmd, None)
    except putils.ProcessExecutionError as err:
        if enforce_multipath:
            LOG.error('multipathd is not running: exit code %(err)s', {'err': err.exit_code})
            raise
        return False
    return True