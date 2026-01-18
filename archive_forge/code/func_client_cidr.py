from pprint import pformat
from six import iteritems
import re
@client_cidr.setter
def client_cidr(self, client_cidr):
    """
        Sets the client_cidr of this V1ServerAddressByClientCIDR.
        The CIDR with which clients can match their IP to figure out the server
        address that they should use.

        :param client_cidr: The client_cidr of this V1ServerAddressByClientCIDR.
        :type: str
        """
    if client_cidr is None:
        raise ValueError('Invalid value for `client_cidr`, must not be `None`')
    self._client_cidr = client_cidr