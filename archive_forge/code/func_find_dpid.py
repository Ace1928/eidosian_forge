import logging
from os_ken.controller import event
from os_ken.lib.dpid import dpid_to_str
from os_ken.base import app_manager
def find_dpid(self, key, value):
    for dpid, conf in self.confs.items():
        if key in conf:
            if conf[key] == value:
                return dpid
    return None