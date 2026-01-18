import logging
from os_ken.controller import event
from os_ken.lib.dpid import dpid_to_str
from os_ken.base import app_manager
def del_key(self, dpid, key):
    del self.confs[dpid][key]
    self.send_event_to_observers(EventConfSwitchDel(dpid, key))