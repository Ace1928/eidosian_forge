import ipaddress
import socket
from openstack import _log
from openstack import exceptions
from openstack import utils
def get_server_default_ip(cloud, server):
    """Get the configured 'default' address

    It is possible in clouds.yaml to configure for a cloud a network that
    is the 'default_interface'. This is the network that should be used
    to talk to instances on the network.

    :param cloud: the cloud we're working with
    :param server: the server dict from which we want to get the default
                   IPv4 address
    :return: a string containing the IPv4 address or None
    """
    ext_net = cloud.get_default_network()
    if ext_net:
        if cloud._local_ipv6 and (not cloud.force_ipv4):
            versions = [6, 4]
        else:
            versions = [4]
        for version in versions:
            ext_ip = get_server_ip(server, key_name=ext_net['name'], version=version, public=True, cloud_public=not cloud.private)
            if ext_ip is not None:
                return ext_ip
    return None