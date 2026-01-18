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
def end_raid(self, device_path: str) -> None:
    raid_exists = self.is_raid_exists(device_path)
    if raid_exists:
        for i in range(10):
            try:
                cmd_out = self.stop_raid(device_path, True)
                if not cmd_out:
                    break
            except Exception:
                time.sleep(1)
        try:
            is_exist = os.path.exists(device_path)
            LOG.debug('[!] is_exist = %s', is_exist)
            if is_exist:
                self.remove_raid(device_path)
                os.remove(device_path)
        except Exception:
            LOG.debug('[!] Exception_stop_raid!')