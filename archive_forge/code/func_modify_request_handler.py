import ssl
import socket
from os_ken import cfg
from os_ken.base import app_manager
from os_ken.lib import hub
from os_ken.lib import ip
from os_ken.services.protocols.ovsdb import client
from os_ken.services.protocols.ovsdb import event
from os_ken.controller import handler
@handler.set_ev_cls(event.EventModifyRequest)
def modify_request_handler(self, ev):
    system_id = ev.system_id
    client_name = client.RemoteOvsdb.instance_name(system_id)
    remote = self._clients.get(client_name)
    if not remote:
        msg = 'Unknown remote system_id %s' % system_id
        self.logger.info(msg)
        rep = event.EventModifyReply(system_id, None, None, msg)
        return self.reply_to_request(ev, rep)
    return remote.modify_request_handler(ev)