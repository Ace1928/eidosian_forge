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
def assemble_raid(self, drives: list[str], md_path: str, read_only: bool) -> bool:
    cmd = ['mdadm', '--assemble', '--run', md_path]
    if read_only:
        cmd.append('-o')
    for i in range(len(drives)):
        cmd.append(drives[i])
    try:
        self.run_mdadm(cmd, True)
    except putils.ProcessExecutionError as ex:
        LOG.warning('[!] Could not _assemble_raid: %s', str(ex))
        raise ex
    return True