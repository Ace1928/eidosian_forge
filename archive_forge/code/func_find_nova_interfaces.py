import ipaddress
import socket
from openstack import _log
from openstack import exceptions
from openstack import utils
def find_nova_interfaces(addresses, ext_tag=None, key_name=None, version=4, mac_addr=None):
    ret = []
    for k, v in iter(addresses.items()):
        if key_name is not None and k != key_name:
            continue
        for interface_spec in v:
            if ext_tag is not None:
                if 'OS-EXT-IPS:type' not in interface_spec:
                    continue
                elif interface_spec['OS-EXT-IPS:type'] != ext_tag:
                    continue
            if mac_addr is not None:
                if 'OS-EXT-IPS-MAC:mac_addr' not in interface_spec:
                    continue
                elif interface_spec['OS-EXT-IPS-MAC:mac_addr'] != mac_addr:
                    continue
            if interface_spec['version'] == version:
                ret.append(interface_spec)
    return ret