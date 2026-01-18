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
def get_host_nqn(system_uuid: Optional[str]=None) -> Optional[str]:
    """Ensure that hostnqn exists, creating if necessary.

    This method tries to return contents from /etc/nvme/hostnqn and if not
    possible then creates the file calling create_hostnqn and passing provided
    system_uuid and returns the contents of the newly created file.

    Method create_hostnqn gives priority to the provided system_uuid parameter
    for the contents of the file over other alternatives it has.
    """
    try:
        with open('/etc/nvme/hostnqn', 'r') as f:
            host_nqn = f.read().strip()
    except IOError:
        host_nqn = priv_nvme.create_hostnqn(system_uuid)
    except Exception:
        host_nqn = None
    return host_nqn