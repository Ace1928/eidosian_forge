import logging
import http.client as http_client
from ..constants import DEFAULT_SWARM_ADDR_POOL, DEFAULT_SWARM_SUBNET_SIZE
from .. import errors
from .. import types
from .. import utils
@utils.minimum_version('1.24')
def join_swarm(self, remote_addrs, join_token, listen_addr='0.0.0.0:2377', advertise_addr=None, data_path_addr=None):
    """
        Make this Engine join a swarm that has already been created.

        Args:
            remote_addrs (:py:class:`list`): Addresses of one or more manager
                nodes already participating in the Swarm to join.
            join_token (string): Secret token for joining this Swarm.
            listen_addr (string): Listen address used for inter-manager
                communication if the node gets promoted to manager, as well as
                determining the networking interface used for the VXLAN Tunnel
                Endpoint (VTEP). Default: ``'0.0.0.0:2377``
            advertise_addr (string): Externally reachable address advertised
                to other nodes. This can either be an address/port combination
                in the form ``192.168.1.1:4567``, or an interface followed by a
                port number, like ``eth0:4567``. If the port number is omitted,
                the port number from the listen address is used. If
                AdvertiseAddr is not specified, it will be automatically
                detected when possible. Default: ``None``
            data_path_addr (string): Address or interface to use for data path
                traffic. For example, 192.168.1.1, or an interface, like eth0.

        Returns:
            ``True`` if the request went through.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
    data = {'RemoteAddrs': remote_addrs, 'ListenAddr': listen_addr, 'JoinToken': join_token, 'AdvertiseAddr': advertise_addr}
    if data_path_addr is not None:
        if utils.version_lt(self._version, '1.30'):
            raise errors.InvalidVersion('Data address path is only available for API version >= 1.30')
        data['DataPathAddr'] = data_path_addr
    url = self._url('/swarm/join')
    response = self._post_json(url, data=data)
    self._raise_for_status(response)
    return True