import logging
import http.client as http_client
from ..constants import DEFAULT_SWARM_ADDR_POOL, DEFAULT_SWARM_SUBNET_SIZE
from .. import errors
from .. import types
from .. import utils
@utils.minimum_version('1.24')
def init_swarm(self, advertise_addr=None, listen_addr='0.0.0.0:2377', force_new_cluster=False, swarm_spec=None, default_addr_pool=None, subnet_size=None, data_path_addr=None, data_path_port=None):
    """
        Initialize a new Swarm using the current connected engine as the first
        node.

        Args:
            advertise_addr (string): Externally reachable address advertised
                to other nodes. This can either be an address/port combination
                in the form ``192.168.1.1:4567``, or an interface followed by a
                port number, like ``eth0:4567``. If the port number is omitted,
                the port number from the listen address is used. If
                ``advertise_addr`` is not specified, it will be automatically
                detected when possible. Default: None
            listen_addr (string): Listen address used for inter-manager
                communication, as well as determining the networking interface
                used for the VXLAN Tunnel Endpoint (VTEP). This can either be
                an address/port combination in the form ``192.168.1.1:4567``,
                or an interface followed by a port number, like ``eth0:4567``.
                If the port number is omitted, the default swarm listening port
                is used. Default: '0.0.0.0:2377'
            force_new_cluster (bool): Force creating a new Swarm, even if
                already part of one. Default: False
            swarm_spec (dict): Configuration settings of the new Swarm. Use
                ``APIClient.create_swarm_spec`` to generate a valid
                configuration. Default: None
            default_addr_pool (list of strings): Default Address Pool specifies
                default subnet pools for global scope networks. Each pool
                should be specified as a CIDR block, like '10.0.0.0/8'.
                Default: None
            subnet_size (int): SubnetSize specifies the subnet size of the
                networks created from the default subnet pool. Default: None
            data_path_addr (string): Address or interface to use for data path
                traffic. For example, 192.168.1.1, or an interface, like eth0.
            data_path_port (int): Port number to use for data path traffic.
                Acceptable port range is 1024 to 49151. If set to ``None`` or
                0, the default port 4789 will be used. Default: None

        Returns:
            (str): The ID of the created node.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
    url = self._url('/swarm/init')
    if swarm_spec is not None and (not isinstance(swarm_spec, dict)):
        raise TypeError('swarm_spec must be a dictionary')
    if default_addr_pool is not None:
        if utils.version_lt(self._version, '1.39'):
            raise errors.InvalidVersion('Address pool is only available for API version >= 1.39')
        if subnet_size is None:
            subnet_size = DEFAULT_SWARM_SUBNET_SIZE
    if subnet_size is not None:
        if utils.version_lt(self._version, '1.39'):
            raise errors.InvalidVersion('Subnet size is only available for API version >= 1.39')
        if default_addr_pool is None:
            default_addr_pool = DEFAULT_SWARM_ADDR_POOL
    data = {'AdvertiseAddr': advertise_addr, 'ListenAddr': listen_addr, 'DefaultAddrPool': default_addr_pool, 'SubnetSize': subnet_size, 'ForceNewCluster': force_new_cluster, 'Spec': swarm_spec}
    if data_path_addr is not None:
        if utils.version_lt(self._version, '1.30'):
            raise errors.InvalidVersion('Data address path is only available for API version >= 1.30')
        data['DataPathAddr'] = data_path_addr
    if data_path_port is not None:
        if utils.version_lt(self._version, '1.40'):
            raise errors.InvalidVersion('Data path port is only available for API version >= 1.40')
        data['DataPathPort'] = data_path_port
    response = self._post_json(url, data=data)
    return self._result(response, json=True)