from os_ken.controller import handler
from os_ken.controller import event
from os_ken.lib import dpid as dpid_lib
from os_ken.lib import mac as mac_lib
from os_ken.lib.packet import vrrp
from os_ken.lib import addrconv
@property
def address_owner(self):
    return self.priority == vrrp.VRRP_PRIORITY_ADDRESS_OWNER