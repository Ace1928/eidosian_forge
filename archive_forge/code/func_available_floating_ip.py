import ipaddress
import time
import warnings
from openstack.cloud import _utils
from openstack.cloud import exc
from openstack.cloud import meta
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
from openstack import proxy
from openstack import utils
from openstack import warnings as os_warnings
def available_floating_ip(self, network=None, server=None):
    """Get a floating IP from a network or a pool.

        Return the first available floating IP or allocate a new one.

        :param network: Name or ID of the network.
        :param server: Server the IP is for if known

        :returns: a (normalized) structure with a floating IP address
                  description.
        """
    if self._use_neutron_floating():
        try:
            f_ips = self._neutron_available_floating_ips(network=network, server=server)
            return f_ips[0]
        except exceptions.NotFoundException as e:
            self.log.debug("Something went wrong talking to neutron API: '%(msg)s'. Trying with Nova.", {'msg': str(e)})
    f_ips = self._normalize_floating_ips(self._nova_available_floating_ips(pool=network))
    return f_ips[0]