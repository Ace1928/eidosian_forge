import collections
import logging
import os_ken.exception as os_ken_exc
from os_ken.base import app_manager
from os_ken.controller import event
def add_remote_dpid(self, dpid, port_no, remote_dpid):
    if port_no in self.dpids[dpid]:
        raise os_ken_exc.PortAlreadyExist(dpid=dpid, port=port_no, network_id=None)
    self._add_remote_dpid(dpid, port_no, remote_dpid)