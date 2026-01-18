import ipaddress
import socket
from openstack import _log
from openstack import exceptions
from openstack import utils
def get_server_private_ip(server, cloud=None):
    """Find the private IP address

    If Neutron is available, search for a port on a network where
    `router:external` is False and `shared` is False. This combination
    indicates a private network with private IP addresses. This port should
    have the private IP.

    If Neutron is not available, or something goes wrong communicating with it,
    as a fallback, try the list of addresses associated with the server dict,
    looking for an IP type tagged as 'fixed' in the network named 'private'.

    Last resort, ignore the IP type and just look for an IP on the 'private'
    network (e.g., Rackspace).
    """
    if cloud and (not cloud.use_internal_network()):
        return None
    fip_ints = find_nova_interfaces(server['addresses'], ext_tag='floating')
    fip_mac = None
    if fip_ints:
        fip_mac = fip_ints[0].get('OS-EXT-IPS-MAC:mac_addr')
    if cloud:
        int_nets = cloud.get_internal_ipv4_networks()
        for int_net in int_nets:
            int_ip = get_server_ip(server, key_name=int_net['name'], ext_tag='fixed', cloud_public=not cloud.private, mac_addr=fip_mac)
            if int_ip is not None:
                return int_ip
        for int_net in int_nets:
            int_ip = get_server_ip(server, key_name=int_net['name'], cloud_public=not cloud.private, mac_addr=fip_mac)
            if int_ip is not None:
                return int_ip
    ip = get_server_ip(server, ext_tag='fixed', key_name='private', mac_addr=fip_mac)
    if ip:
        return ip
    return get_server_ip(server, key_name='private')