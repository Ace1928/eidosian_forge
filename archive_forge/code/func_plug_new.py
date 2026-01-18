import abc
from neutron_lib import constants
@abc.abstractmethod
def plug_new(self, network_id, port_id, device_name, mac_address, bridge=None, namespace=None, prefix=None, mtu=None, link_up=True):
    """Plug in the interface only for new devices that don't exist yet."""