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
@staticmethod
def get_md_name(device_name: str) -> Optional[str]:
    try:
        with open('/proc/mdstat', 'r') as f:
            lines = [line.split(' ')[0] for line in f if device_name in line]
            if lines:
                return lines[0]
    except Exception as exc:
        LOG.debug('[!] Could not find md name for %s in mdstat: %s', device_name, exc)
    return None