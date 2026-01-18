import uuid
from os_ken.lib import dpid as dpidlib
from os_ken.services.protocols.ovsdb import event as ovsdb_event
def set_external_id(manager, system_id, key, val, fn):
    val = str(val)

    def _set_iface_external_id(tables, *_):
        row = fn(tables)
        if not row:
            return None
        external_ids = row.external_ids
        external_ids[key] = val
        row.external_ids = external_ids
    req = ovsdb_event.EventModifyRequest(system_id, _set_iface_external_id)
    return manager.send_request(req)