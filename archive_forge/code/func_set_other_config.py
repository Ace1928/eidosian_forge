import uuid
from os_ken.lib import dpid as dpidlib
from os_ken.services.protocols.ovsdb import event as ovsdb_event
def set_other_config(manager, system_id, key, val, fn):
    val = str(val)

    def _set_iface_other_config(tables, *_):
        row = fn(tables)
        if not row:
            return None
        other_config = row.other_config
        other_config[key] = val
        row.other_config = other_config
    req = ovsdb_event.EventModifyRequest(system_id, _set_iface_other_config)
    return manager.send_request(req)