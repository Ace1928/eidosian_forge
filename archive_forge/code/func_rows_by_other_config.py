import uuid
from os_ken.lib import dpid as dpidlib
from os_ken.services.protocols.ovsdb import event as ovsdb_event
def rows_by_other_config(manager, system_id, key, value, table='Bridge', fn=None):
    matched_rows = match_rows(manager, system_id, table, lambda r: key in r.other_config and r.other_config.get(key) == value)
    if matched_rows and fn is not None:
        return [fn(row) for row in matched_rows]
    return matched_rows