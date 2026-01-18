import uuid
from os_ken.lib import dpid as dpidlib
from os_ken.services.protocols.ovsdb import event as ovsdb_event
def get_bridge_by_datapath_id(manager, system_id, datapath_id, fn=None):

    def _match_fn(row):
        row_dpid = dpidlib.str_to_dpid(str(row.datapath_id[0]))
        return row_dpid == datapath_id
    bridge = match_row(manager, system_id, 'Bridge', _match_fn)
    if fn is not None:
        return fn(bridge)
    return bridge