import ssl
import socket
from os_ken import cfg
from os_ken.base import app_manager
from os_ken.lib import hub
from os_ken.lib import ip
from os_ken.services.protocols.ovsdb import client
from os_ken.services.protocols.ovsdb import event
from os_ken.controller import handler
@handler.set_ev_cls(event.EventReadRequest)
def read_request_handler(self, ev):
    system_id = ev.system_id
    if system_id is None:

        def done(gt, *args, **kwargs):
            if gt in self.threads:
                self.threads.remove(gt)
        thread = hub.spawn(self._bulk_read_handler, ev)
        self.threads.append(thread)
        return thread.link(done)
    client_name = client.RemoteOvsdb.instance_name(system_id)
    remote = self._clients.get(client_name)
    if not remote:
        msg = 'Unknown remote system_id %s' % system_id
        self.logger.info(msg)
        rep = event.EventReadReply(system_id, None, msg)
        return self.reply_to_request(ev, rep)
    return remote.read_request_handler(ev)