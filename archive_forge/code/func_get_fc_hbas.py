from __future__ import annotations
import glob
import os
from typing import Iterable
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick.initiator import linuxscsi
@classmethod
def get_fc_hbas(cls) -> list[dict[str, str]]:
    """Get the Fibre Channel HBA information from sysfs."""
    hbas = []
    for hostpath in glob.glob(f'{cls.FC_HOST_SYSFS_PATH}/*'):
        try:
            hba = {'ClassDevice': os.path.basename(hostpath), 'ClassDevicepath': os.path.realpath(hostpath)}
            for attribute in cls.HBA_ATTRIBUTES:
                with open(os.path.join(hostpath, attribute), 'rt') as f:
                    hba[attribute] = f.read().strip()
            hbas.append(hba)
        except Exception as exc:
            LOG.warning('Could not read attributes for %(hp)s: %(exc)s', {'hp': hostpath, 'exc': exc})
    return hbas