import os
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick import caches
from os_brick import exception
from os_brick import executor
def os_execute(self, *cmd, **kwargs):
    LOG.debug('os_execute: cmd: %s, args: %s', cmd, kwargs)
    try:
        out, err = self._execute(*cmd, **kwargs)
    except putils.ProcessExecutionError as err:
        LOG.exception('os_execute error')
        LOG.error('Cmd     :%s', err.cmd)
        LOG.error('StdOut  :%s', err.stdout)
        LOG.error('StdErr  :%s', err.stderr)
        raise
    return (out, err)