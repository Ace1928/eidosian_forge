import collections
from os_ken.base import app_manager
import os_ken.exception as os_ken_exc
from os_ken.controller import event
from os_ken.exception import NetworkNotFound, NetworkAlreadyExist
from os_ken.exception import PortAlreadyExist, PortNotFound, PortUnknown
def remove_port(self, network_id, dpid, port_no):
    old_mac_address = self._get_old_mac(network_id, dpid, port_no)
    self.dpids.remove_port(dpid, port_no)
    try:
        self.networks.remove(network_id, dpid, port_no)
    except NetworkNotFound:
        pass
    if old_mac_address is not None:
        self.mac_addresses.remove_port(network_id, dpid, port_no, old_mac_address)