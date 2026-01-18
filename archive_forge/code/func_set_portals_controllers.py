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
def set_portals_controllers(self) -> None:
    """Search and set controller names in the target's portals.

        Compare the address, port, and transport protocol for each portal
        against existing nvme subsystem controllers.
        """
    if all((p.controller for p in self.portals)):
        return
    hostnqn: Optional[str] = self.host_nqn or utils.get_host_nqn()
    sysfs_portals: list[tuple[Optional[str], Optional[str], Optional[Union[str, utils.Anything]], Optional[Union[str, utils.Anything]]]] = [(p.address, p.port, p.transport, hostnqn) for p in self.portals]
    known_names: list[str] = [p.controller for p in self.portals if p.controller]
    warned = False
    LOG.debug('Search controllers for portals %s', sysfs_portals)
    ctrl_paths = glob.glob(NVME_CTRL_SYSFS_PATH + 'nvme*')
    for ctrl_path in ctrl_paths:
        ctrl_name = os.path.basename(ctrl_path)
        if ctrl_name in known_names:
            continue
        LOG.debug('Checking controller %s', ctrl_name)
        nqn = sysfs_property('subsysnqn', ctrl_path)
        if nqn != self.nqn:
            LOG.debug("Skipping %s, doesn't match %s", nqn, self.nqn)
            continue
        ctrl_transport = sysfs_property('transport', ctrl_path)
        address = sysfs_property('address', ctrl_path)
        if not address:
            LOG.error("Couldn't read address for %s", ctrl_path)
            continue
        ctrl_address = dict((x.split('=') for x in address.split(',')))
        ctrl_addr = ctrl_address['traddr']
        ctrl_port = ctrl_address['trsvcid']
        ctrl_hostnqn = sysfs_property('hostnqn', ctrl_path) or utils.ANY
        if ctrl_hostnqn is utils.ANY and (not warned):
            LOG.warning("OS doesn't present the host nqn information. Controller may be incorrectly matched.")
            warned = True
        ctrl_portal = (ctrl_addr, ctrl_port, ctrl_transport, ctrl_hostnqn)
        try:
            index = sysfs_portals.index(ctrl_portal)
            LOG.debug('Found a valid portal at %s', ctrl_portal)
            self.portals[index].controller = ctrl_name
            known_names.append(ctrl_name)
        except ValueError:
            LOG.debug('Skipping %s, not part of portals %s', ctrl_portal, sysfs_portals)
        if len(known_names) == len(sysfs_portals):
            return