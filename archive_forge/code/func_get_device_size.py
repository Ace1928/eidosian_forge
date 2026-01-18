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
def get_device_size(executor: executor.Executor, device: str) -> Optional[int]:
    """Get the size in bytes of a volume."""
    out, _err = executor._execute('blockdev', '--getsize64', device, run_as_root=True, root_helper=executor._root_helper)
    var = str(out.strip())
    if var.isnumeric():
        return int(var)
    else:
        return None