from __future__ import annotations
import functools
import inspect
import logging as py_logging
import os
import time
from typing import Any, Callable, Optional, Type, Union   # noqa: H301
import uuid as uuid_lib
from oslo_concurrency import processutils
from oslo_log import log as logging
from oslo_utils import strutils
from os_brick import executor
from os_brick.i18n import _
from os_brick.privileged import nvmeof as priv_nvme
from os_brick.privileged import rootwrap as priv_rootwrap
import tenacity  # noqa
def get_nvme_host_id(uuid: Optional[str]) -> Optional[str]:
    """Get the nvme host id

    If the hostid file doesn't exist create it either with the passed uuid or
    a random one.
    """
    try:
        with open('/etc/nvme/hostid', 'r') as f:
            host_id = f.read().strip()
    except IOError:
        uuid = uuid or str(uuid_lib.uuid4())
        host_id = priv_nvme.create_hostid(uuid)
    except Exception:
        host_id = None
    return host_id