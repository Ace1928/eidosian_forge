import uuid
from os_ken.lib import dpid as dpidlib
from os_ken.services.protocols.ovsdb import event as ovsdb_event
def get_bridge_for_iface_name(manager, system_id, iface_name, fn=None):
    iface = row_by_name(manager, system_id, iface_name, 'Interface')
    port = match_row(manager, system_id, 'Port', lambda x: iface in x.interfaces)
    bridge = match_row(manager, system_id, 'Bridge', lambda x: port in x.ports)
    if fn is not None:
        return fn(bridge)
    return bridge