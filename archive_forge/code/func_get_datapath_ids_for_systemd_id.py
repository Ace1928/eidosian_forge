import uuid
from os_ken.lib import dpid as dpidlib
from os_ken.services.protocols.ovsdb import event as ovsdb_event
def get_datapath_ids_for_systemd_id(manager, system_id):

    def _get_dp_ids(tables):
        dp_ids = []
        bridges = tables.get('Bridge')
        if not bridges:
            return dp_ids
        for bridge in bridges.rows.values():
            datapath_ids = bridge.datapath_id
            dp_ids.extend((dpidlib.str_to_dpid(dp_id) for dp_id in datapath_ids))
        return dp_ids
    request = ovsdb_event.EventReadRequest(system_id, _get_dp_ids)
    reply = manager.send_request(request)
    return reply.result