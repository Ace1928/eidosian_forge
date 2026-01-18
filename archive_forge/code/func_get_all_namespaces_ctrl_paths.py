from __future__ import annotations
import errno
import functools
import glob
import json
import os.path
import time
from typing import (Callable, Optional, Sequence, Type, Union)  # noqa: H301
import uuid as uuid_lib
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator.connectors import base
from os_brick.privileged import nvmeof as priv_nvmeof
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick import utils
def get_all_namespaces_ctrl_paths(self) -> list[str]:
    """Return all nvme sysfs control paths for this portal.

        The basename of the path can be single volume or a channel to an ANA
        volume.

        For example for the nvme1 controller we could return:

            ['/sys/class/nvme-fabrics/ctl/nvme1n1 ',
             '/sys/class/nvme-fabrics/ctl/nvme0c1n1']
        """
    if not self.controller:
        return []
    return glob.glob(f'{NVME_CTRL_SYSFS_PATH}{self.controller}/nvme*')