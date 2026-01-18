import uuid
from os_ken.lib import dpid as dpidlib
from os_ken.services.protocols.ovsdb import event as ovsdb_event
def get_iface_by_name(manager, system_id, name, fn=None):
    iface = row_by_name(manager, system_id, name, 'Interface')
    if fn is not None:
        return fn(iface)
    return iface