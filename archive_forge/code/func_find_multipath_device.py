from __future__ import annotations
import glob
import os
import re
import time
from typing import Optional
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_utils import excutils
from os_brick import constants
from os_brick import exception
from os_brick import executor
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick import utils
def find_multipath_device(self, device):
    """Discover multipath devices for a mpath device.

           This uses the slow multipath -l command to find a
           multipath device description, then screen scrapes
           the output to discover the multipath device name
           and it's devices.

        """
    mdev = None
    devices = []
    out = None
    try:
        out, _err = self._execute('multipath', '-l', device, run_as_root=True, root_helper=self._root_helper)
    except putils.ProcessExecutionError as exc:
        LOG.warning('multipath call failed exit %(code)s', {'code': exc.exit_code})
        raise exception.CommandExecutionFailed(cmd='multipath -l %s' % device)
    if out:
        lines_str = out.strip()
        lines = lines_str.split('\n')
        lines = [line for line in lines if not re.match(MULTIPATH_ERROR_REGEX, line) and len(line)]
        if lines:
            mdev_name = lines[0].split(' ')[0]
            if mdev_name in MULTIPATH_DEVICE_ACTIONS:
                mdev_name = lines[0].split(' ')[1]
            mdev = '/dev/mapper/%s' % mdev_name
            try:
                os.stat(mdev)
            except OSError:
                LOG.warning("Couldn't find multipath device %s", mdev)
                return None
            wwid_search = MULTIPATH_WWID_REGEX.search(lines[0])
            if wwid_search is not None:
                mdev_id = wwid_search.group('wwid')
            else:
                mdev_id = mdev_name
            LOG.debug('Found multipath device = %(mdev)s', {'mdev': mdev})
            device_lines = lines[3:]
            for dev_line in device_lines:
                if dev_line.find('policy') != -1:
                    continue
                dev_line = dev_line.lstrip(' |-`')
                dev_info = dev_line.split()
                address = dev_info[0].split(':')
                dev = {'device': '/dev/%s' % dev_info[1], 'host': address[0], 'channel': address[1], 'id': address[2], 'lun': address[3]}
                devices.append(dev)
    if mdev is not None:
        info = {'device': mdev, 'id': mdev_id, 'name': mdev_name, 'devices': devices}
        return info
    return None