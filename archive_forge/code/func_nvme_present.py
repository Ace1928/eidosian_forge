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
@classmethod
def nvme_present(cls: type) -> bool:
    """Check if the nvme CLI is present."""
    try:
        priv_rootwrap.custom_execute('nvme', 'version')
        return True
    except Exception as exc:
        if isinstance(exc, OSError) and exc.errno == errno.ENOENT:
            LOG.debug('nvme not present on system')
        else:
            LOG.warning('Unknown error when checking presence of nvme: %s', exc)
    return False