import uuid
from os_ken.lib import dpid as dpidlib
from os_ken.services.protocols.ovsdb import event as ovsdb_event
def row_by_name(manager, system_id, name, table='Bridge', fn=None):
    matched_row = match_row(manager, system_id, table, lambda row: row.name == name)
    if fn is not None:
        return fn(matched_row)
    return matched_row