import collections
import logging
import os_ken.exception as os_ken_exc
from os_ken.base import app_manager
from os_ken.controller import event
def get_remote_dpid(self, dpid, port_no):
    return self.dpids.get_remote_dpid(dpid, port_no)