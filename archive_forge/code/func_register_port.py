import collections
import logging
import os_ken.exception as os_ken_exc
from os_ken.base import app_manager
from os_ken.controller import event
def register_port(self, dpid, port_no, remote_dpid):
    self.dpids.add_remote_dpid(dpid, port_no, remote_dpid)