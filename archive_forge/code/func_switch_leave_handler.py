import logging
import time
from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.topology import event
from os_ken.topology import switches
@handler.set_ev_cls(event.EventSwitchLeave)
def switch_leave_handler(self, ev):
    LOG.debug(ev)