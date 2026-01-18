import uuid
from os_ken.lib import dpid as dpidlib
from os_ken.services.protocols.ovsdb import event as ovsdb_event
def match_rows(manager, system_id, table, fn):

    def _match_rows(tables):
        return (r for r in tables[table].rows.values() if fn(r))
    request = ovsdb_event.EventReadRequest(system_id, _match_rows)
    reply = manager.send_request(request)
    return reply.result