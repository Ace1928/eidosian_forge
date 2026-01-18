from ..errors import InvalidVersion
from ..utils import check_resource, minimum_version
from ..utils import version_lt
from .. import utils
@check_resource('container')
def connect_container_to_network(self, container, net_id, ipv4_address=None, ipv6_address=None, aliases=None, links=None, link_local_ips=None, driver_opt=None, mac_address=None):
    """
        Connect a container to a network.

        Args:
            container (str): container-id/name to be connected to the network
            net_id (str): network id
            aliases (:py:class:`list`): A list of aliases for this endpoint.
                Names in that list can be used within the network to reach the
                container. Defaults to ``None``.
            links (:py:class:`list`): A list of links for this endpoint.
                Containers declared in this list will be linked to this
                container. Defaults to ``None``.
            ipv4_address (str): The IP address of this container on the
                network, using the IPv4 protocol. Defaults to ``None``.
            ipv6_address (str): The IP address of this container on the
                network, using the IPv6 protocol. Defaults to ``None``.
            link_local_ips (:py:class:`list`): A list of link-local
                (IPv4/IPv6) addresses.
            mac_address (str): The MAC address of this container on the
                network. Defaults to ``None``.
        """
    data = {'Container': container, 'EndpointConfig': self.create_endpoint_config(aliases=aliases, links=links, ipv4_address=ipv4_address, ipv6_address=ipv6_address, link_local_ips=link_local_ips, driver_opt=driver_opt, mac_address=mac_address)}
    url = self._url('/networks/{0}/connect', net_id)
    res = self._post_json(url, data=data)
    self._raise_for_status(res)