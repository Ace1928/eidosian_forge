import ipaddress
import socket
from openstack import _log
from openstack import exceptions
from openstack import utils
def find_nova_addresses(addresses, ext_tag=None, key_name=None, version=4, mac_addr=None):
    interfaces = find_nova_interfaces(addresses, ext_tag, key_name, version, mac_addr)
    floating_addrs = []
    fixed_addrs = []
    for i in interfaces:
        if i.get('OS-EXT-IPS:type') == 'floating':
            floating_addrs.append(i['addr'])
        else:
            fixed_addrs.append(i['addr'])
    return floating_addrs + fixed_addrs