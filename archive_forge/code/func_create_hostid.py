from __future__ import annotations
import errno
import os
from typing import Optional  # noqa: H301
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
import os_brick.privileged
from os_brick.privileged import rootwrap
@os_brick.privileged.default.entrypoint
def create_hostid(uuid: str) -> Optional[str]:
    """Create the hostid to ensure it's always the same."""
    try:
        os.makedirs('/etc/nvme', mode=493, exist_ok=True)
        with open('/etc/nvme/hostid', 'w') as f:
            LOG.debug('Writing nvme hostid %s', uuid)
            f.write(f'{uuid}\n')
        os.chmod('/etc/nvme/hostid', 420)
    except Exception as e:
        LOG.warning('Could not generate nvme host id: %s', e)
        return None
    return uuid