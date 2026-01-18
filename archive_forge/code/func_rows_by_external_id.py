import uuid
from os_ken.lib import dpid as dpidlib
from os_ken.services.protocols.ovsdb import event as ovsdb_event
def rows_by_external_id(manager, system_id, key, value, table='Bridge', fn=None):
    matched_rows = match_rows(manager, system_id, table, lambda r: key in r.external_ids and r.external_ids.get(key) == value)
    if matched_rows and fn is not None:
        return [fn(row) for row in matched_rows]
    return matched_rows