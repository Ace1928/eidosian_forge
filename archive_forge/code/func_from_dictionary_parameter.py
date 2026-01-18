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
def from_dictionary_parameter(cls: Type['NVMeOFConnProps'], func: Callable) -> Callable:
    """Decorator to convert connection properties dictionary.

        It converts the connection properties into a NVMeOFConnProps instance
        and finds the controller names for all portals present in the system.
        """

    @functools.wraps(func)
    def wrapper(self, connection_properties, *args, **kwargs):
        conn_props = cls(connection_properties, find_controllers=True)
        return func(self, conn_props, *args, **kwargs)
    return wrapper