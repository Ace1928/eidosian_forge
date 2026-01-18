from __future__ import annotations
import glob
import os
from typing import Iterable
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick.initiator import linuxscsi
def get_fc_wwpns(self) -> list[str]:
    """Get Fibre Channel WWPNs from the system, if any."""
    hbas = self.get_fc_hbas()
    wwpns = []
    for hba in hbas:
        if hba['port_state'] == 'Online':
            wwpn = hba['port_name'].replace('0x', '')
            wwpns.append(wwpn)
    return wwpns