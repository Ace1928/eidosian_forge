from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def community_maps_tmplt(config_data):
    name = config_data.get('name', '')
    command = 'snmp-server community-map {name}'.format(name=name)
    if config_data.get('context'):
        command += ' context {context}'.format(context=config_data['context'])
    if config_data.get('security_name'):
        command += ' security-name {security_name}'.format(security_name=config_data['security_name'])
    if config_data.get('target_list'):
        command += ' target-list {target_list}'.format(target_list=config_data['target_list'])
    return command