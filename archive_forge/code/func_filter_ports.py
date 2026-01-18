import collections
from os_ken.base import app_manager
import os_ken.exception as os_ken_exc
from os_ken.controller import event
from os_ken.exception import NetworkNotFound, NetworkAlreadyExist
from os_ken.exception import PortAlreadyExist, PortNotFound, PortUnknown
def filter_ports(self, dpid, in_port, nw_id, allow_nw_id_external=None):
    assert nw_id != self.nw_id_unknown
    ret = []
    for port in self.get_ports(dpid):
        nw_id_ = port.network_id
        if port.port_no == in_port:
            continue
        if nw_id_ == nw_id:
            ret.append(port.port_no)
        elif allow_nw_id_external is not None and nw_id_ == allow_nw_id_external:
            ret.append(port.port_no)
    return ret