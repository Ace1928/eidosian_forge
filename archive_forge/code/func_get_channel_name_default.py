from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config, set_nc_config, check_ip_addr
def get_channel_name_default(channel_id):
    """get default out direct"""
    channel_dict = {'0': 'console', '1': 'monitor', '2': 'loghost', '3': 'trapbuffer', '4': 'logbuffer', '5': 'snmpagent', '6': 'channel6', '7': 'channel7', '8': 'channel8', '9': 'channel9'}
    channel_name_default = channel_dict.get(channel_id)
    return channel_name_default