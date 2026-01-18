from __future__ import annotations
import errno
import os
from typing import Optional  # noqa: H301
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
import os_brick.privileged
from os_brick.privileged import rootwrap
@os_brick.privileged.default.entrypoint
def get_system_uuid() -> str:
    try:
        with open('/sys/class/dmi/id/product_uuid', 'r') as f:
            return f.read().strip()
    except Exception:
        LOG.debug("Could not read dmi's 'product_uuid' on sysfs")
    try:
        out, err = rootwrap.custom_execute('dmidecode', '-ssystem-uuid')
        if not out:
            LOG.warning('dmidecode returned empty system-uuid')
    except (putils.ProcessExecutionError, FileNotFoundError) as e:
        LOG.debug('Unable to locate dmidecode. For Cinder RSD Backend, please make sure it is installed: %s', e)
        out = ''
    return out.strip()