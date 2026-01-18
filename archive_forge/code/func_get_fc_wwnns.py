from __future__ import annotations
import glob
import os
from typing import Iterable
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick.initiator import linuxscsi
def get_fc_wwnns(self) -> list[str]:
    """Get Fibre Channel WWNNs from the system, if any."""
    hbas = self.get_fc_hbas()
    wwnns = []
    for hba in hbas:
        if hba['port_state'] == 'Online':
            wwnn = hba['node_name'].replace('0x', '')
            wwnns.append(wwnn)
    return wwnns