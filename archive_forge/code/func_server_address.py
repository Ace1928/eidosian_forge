from pprint import pformat
from six import iteritems
import re
@server_address.setter
def server_address(self, server_address):
    """
        Sets the server_address of this V1ServerAddressByClientCIDR.
        Address of this server, suitable for a client that matches the above
        CIDR. This can be a hostname, hostname:port, IP or IP:port.

        :param server_address: The server_address of this
        V1ServerAddressByClientCIDR.
        :type: str
        """
    if server_address is None:
        raise ValueError('Invalid value for `server_address`, must not be `None`')
    self._server_address = server_address