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
def run_mdadm(self, cmd: Sequence[str], raise_exception: bool=False) -> Optional[str]:
    cmd_output = None
    try:
        lines, err = self._execute(*cmd, run_as_root=True, root_helper=self._root_helper)
        for line in lines.split('\n'):
            cmd_output = line
            break
    except putils.ProcessExecutionError as ex:
        LOG.warning('[!] Could not run mdadm: %s', str(ex))
        if raise_exception:
            raise ex
    return cmd_output