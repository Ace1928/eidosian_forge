import collections
from os_ken.base import app_manager
import os_ken.exception as os_ken_exc
from os_ken.controller import event
from os_ken.exception import NetworkNotFound, NetworkAlreadyExist
from os_ken.exception import PortAlreadyExist, PortNotFound, PortUnknown
def set_mac(self, network_id, dpid, port_no, mac_address):
    port = self.get_port(dpid, port_no)
    if port.mac_address is not None:
        raise MacAddressAlreadyExist(dpid=dpid, port=port_no, mac_address=mac_address)
    self._set_mac(network_id, dpid, port_no, port, mac_address)