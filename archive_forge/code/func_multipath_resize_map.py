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
def multipath_resize_map(self, dm_path):
    """Issue a multipath resize map on device.

        This forces the multipath daemon to update it's
        size information a particular multipath device.

        :param dm_path: Real path of the DM device (eg: /dev/dm-5)
        """
    tstart = time.time()
    while True:
        try:
            self._multipath_resize_map(dm_path)
            break
        except putils.ProcessExecutionError as err:
            with excutils.save_and_reraise_exception(reraise=True) as ctx:
                elapsed = time.time() - tstart
                if 'timeout' in err.stdout and elapsed < MULTIPATHD_RESIZE_TIMEOUT:
                    LOG.debug('multipathd resize map timed out. Elapsed: %s, timeout: %s. Retrying...', elapsed, MULTIPATHD_RESIZE_TIMEOUT)
                    ctx.reraise = False
                    time.sleep(1)