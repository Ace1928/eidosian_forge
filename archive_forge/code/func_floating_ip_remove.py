from keystoneauth1 import exceptions as ksa_exceptions
from osc_lib.api import api
from osc_lib import exceptions
from osc_lib.i18n import _
def floating_ip_remove(self, server, address):
    """Remove a floating IP from a server

        :param server:
            The :class:`Server` (or its ID) to add an IP to.
        :param address:
            The FloatingIP or string floating address to add.
        """
    url = '/servers'
    server = self.find(url, attr='name', value=server)
    address = address.ip if hasattr(address, 'ip') else address
    body = {'address': address}
    return self._request('POST', '/%s/%s/action' % (url, server['id']), json={'removeFloatingIp': body})