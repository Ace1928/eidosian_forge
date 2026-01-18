import logging
from os_ken.exception import MacAddressDuplicated
from os_ken.lib.mac import haddr_to_str
def del_mac(self, mac):
    del self.mac_to_net[mac]