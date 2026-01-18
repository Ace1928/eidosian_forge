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
@functools.wraps(func)
def change_encrypted(*args, **kwargs):
    call_args = inspect.getcallargs(func, *args, **kwargs)
    conn_props = call_args['connection_properties']
    custom_symlink = False
    if conn_props.get('encrypted'):
        dev_info = call_args.get('device_info')
        symlink = get_dev_path(conn_props, dev_info)
        devpath = _device_path_from_symlink(symlink)
        if isinstance(symlink, str) and symlink != devpath:
            custom_symlink = True
            call_args['connection_properties'] = conn_props.copy()
            call_args['connection_properties']['device_path'] = devpath
            if dev_info:
                dev_info = call_args['device_info'] = dev_info.copy()
                dev_info['path'] = devpath
    res = func(**call_args)
    if custom_symlink and unlink_after:
        try:
            priv_rootwrap.unlink_root(symlink)
        except Exception:
            LOG.warning('Failed to remove encrypted custom symlink %s', symlink)
    return res