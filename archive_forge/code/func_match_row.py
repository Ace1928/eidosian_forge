import uuid
from os_ken.lib import dpid as dpidlib
from os_ken.services.protocols.ovsdb import event as ovsdb_event
def match_row(manager, system_id, table, fn):

    def _match_row(tables):
        return next((r for r in tables[table].rows.values() if fn(r)), None)
    request_to_get_tables = ovsdb_event.EventReadRequest(system_id, _match_row)
    reply_to_get_tables = manager.send_request(request_to_get_tables)
    return reply_to_get_tables.result