import logging
import time
from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.topology import event
from os_ken.topology import switches
@handler.set_ev_cls(event.EventSwitchReply)
def switch_reply_handler(self, reply):
    LOG.debug('switch_reply async %s', reply)
    if len(reply.switches) > 0:
        for sw in reply.switches:
            LOG.debug('  %s', sw)