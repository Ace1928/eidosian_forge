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
def add_ip_list(self, server, ips, wait=False, timeout=60, fixed_address=None, nat_destination=None):
    """Attach a list of IPs to a server.

        :param server: a server object
        :param ips: list of floating IP addresses or a single address
        :param wait: (optional) Wait for the address to appear as assigned
                     to the server. Defaults to False.
        :param timeout: (optional) Seconds to wait, defaults to 60.
                        See the ``wait`` parameter.
        :param fixed_address: (optional) Fixed address of the server to
                                         attach the IP to
        :param nat_destination: (optional) Name or ID of the network that
                                          the fixed IP to attach the
                                          floating IP should be on

        :returns: The updated server ``openstack.compute.v2.server.Server``
        :raises: :class:`~openstack.exceptions.SDKException` on operation
            error.
        """
    if type(ips) is list:
        ips = [ips]
    for ip in ips:
        f_ip = self.get_floating_ip(id=None, filters={'floating_ip_address': ip})
        server = self._attach_ip_to_server(server=server, floating_ip=f_ip, wait=wait, timeout=timeout, fixed_address=fixed_address, nat_destination=nat_destination)
    return server