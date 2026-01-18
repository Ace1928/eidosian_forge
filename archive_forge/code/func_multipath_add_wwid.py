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
def multipath_add_wwid(self, wwid):
    """Add a wwid to the list of know multipath wwids.

        This has the effect of multipathd being willing to create a dm for a
        multipath even when there's only 1 device.
        """
    out, err = self._execute('multipath', '-a', wwid, run_as_root=True, check_exit_code=False, root_helper=self._root_helper)
    return out.strip() == "wwid '" + wwid + "' added"