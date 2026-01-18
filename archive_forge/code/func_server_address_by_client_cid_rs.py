from pprint import pformat
from six import iteritems
import re
@server_address_by_client_cid_rs.setter
def server_address_by_client_cid_rs(self, server_address_by_client_cid_rs):
    """
        Sets the server_address_by_client_cid_rs of this V1APIGroup.
        a map of client CIDR to server address that is serving this group. This
        is to help clients reach servers in the most network-efficient way
        possible. Clients can use the appropriate server address as per the CIDR
        that they match. In case of multiple matches, clients should use the
        longest matching CIDR. The server returns only those CIDRs that it
        thinks that the client can match. For example: the master will return an
        internal IP CIDR only, if the client reaches the server using an
        internal IP. Server looks at X-Forwarded-For header or X-Real-Ip header
        or request.RemoteAddr (in that order) to get the client IP.

        :param server_address_by_client_cid_rs: The
        server_address_by_client_cid_rs of this V1APIGroup.
        :type: list[V1ServerAddressByClientCIDR]
        """
    self._server_address_by_client_cid_rs = server_address_by_client_cid_rs