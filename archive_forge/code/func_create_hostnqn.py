from __future__ import annotations
import errno
import os
from typing import Optional  # noqa: H301
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
import os_brick.privileged
from os_brick.privileged import rootwrap
@os_brick.privileged.default.entrypoint
def create_hostnqn(system_uuid: Optional[str]=None) -> str:
    """Create the hostnqn file to speed up finding out the nqn.

    By having the /etc/nvme/hostnqn not only do we make sure that that value is
    always used on this system, but we are also able to just open the file to
    get the nqn on each get_connector_properties call instead of having to make
    a call to nvme show-hostnqn command.

    In newer nvme-cli versions calling show-hostnqn will not only try to
    locate the file (which we know doesn't exist or this method wouldn't have
    been called), but it will also generate one.  In older nvme-cli versions
    that is not the case.
    """
    host_nqn = ''
    try:
        os.makedirs('/etc/nvme', mode=493, exist_ok=True)
        if system_uuid:
            host_nqn = 'nqn.2014-08.org.nvmexpress:uuid:' + system_uuid
        else:
            try:
                host_nqn, err = rootwrap.custom_execute('nvme', 'show-hostnqn')
                host_nqn = host_nqn.strip()
            except putils.ProcessExecutionError as e:
                err_msg = e.stdout + '\n' + e.stderr
                msg = err_msg.casefold()
                if 'error: invalid sub-command' in msg:
                    LOG.debug('Version too old cannot check current hostnqn.')
                elif 'hostnqn is not available' in msg:
                    LOG.debug('Version too old to return hostnqn from non file sources')
                elif e.exit_code == errno.ENOENT:
                    LOG.debug('No nqn could be formed from dmi or systemd.')
                else:
                    LOG.debug('Unknown error from nvme show-hostnqn: %s', err_msg)
                    raise
            if not host_nqn:
                LOG.debug('Generating nqn')
                host_nqn, err = rootwrap.custom_execute('nvme', 'gen-hostnqn')
                host_nqn = host_nqn.strip()
        with open('/etc/nvme/hostnqn', 'w') as f:
            LOG.debug('Writing hostnqn file')
            f.write(host_nqn)
        os.chmod('/etc/nvme/hostnqn', 420)
    except Exception as e:
        LOG.warning('Could not generate host nqn: %s', e)
    return host_nqn