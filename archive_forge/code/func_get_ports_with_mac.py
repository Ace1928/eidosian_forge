import collections
from os_ken.base import app_manager
import os_ken.exception as os_ken_exc
from os_ken.controller import event
from os_ken.exception import NetworkNotFound, NetworkAlreadyExist
from os_ken.exception import PortAlreadyExist, PortNotFound, PortUnknown
def get_ports_with_mac(self, network_id, mac_address):
    return self.mac_addresses.get_ports(network_id, mac_address)