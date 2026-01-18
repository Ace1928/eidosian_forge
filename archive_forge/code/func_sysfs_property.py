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
def sysfs_property(prop_name: str, path_or_name: str) -> Optional[str]:
    """Get sysfs property by path returning None if not possible."""
    filename = os.path.join(path_or_name, prop_name)
    LOG.debug('Checking property at %s', filename)
    try:
        with open(filename, 'r') as f:
            result = f.read().strip()
            LOG.debug('Contents: %s', result)
            return result
    except (FileNotFoundError, IOError) as exc:
        LOG.debug('Error reading file %s', exc)
        return None