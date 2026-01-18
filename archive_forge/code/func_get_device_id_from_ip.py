from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
def get_device_id_from_ip(ip_addresses, device_list, module):
    ip_map = dict([(each_device['DeviceManagement'][0]['NetworkAddress'], each_device['Id']) for each_device in device_list if each_device['DeviceManagement']])
    device_id_list_map = {}
    for available_ip, device_id in ip_map.items():
        for ip_formats in ip_addresses:
            if isinstance(ip_formats, IPAddress):
                try:
                    ome_ip = IPAddress(available_ip)
                except AddrFormatError:
                    ome_ip = IPAddress(available_ip.replace(']', '').replace('[', ''))
                if ome_ip == ip_formats:
                    device_id_list_map.update({device_id: str(ip_formats)})
            if not isinstance(ip_formats, IPAddress):
                try:
                    ome_ip = IPAddress(available_ip)
                except AddrFormatError:
                    ome_ip = IPAddress(available_ip.replace(']', '').replace('[', ''))
                if ome_ip in ip_formats:
                    device_id_list_map.update({device_id: str(ome_ip)})
    if len(device_id_list_map) == 0:
        module.fail_json(msg=IP_NOT_EXISTS)
    return device_id_list_map