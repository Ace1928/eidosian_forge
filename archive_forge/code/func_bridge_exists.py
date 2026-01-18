import uuid
from os_ken.lib import dpid as dpidlib
from os_ken.services.protocols.ovsdb import event as ovsdb_event
def bridge_exists(manager, system_id, bridge_name):
    return bool(row_by_name(manager, system_id, bridge_name))