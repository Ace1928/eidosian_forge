from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def group_tmplt(config_data):
    group = config_data.get('group', '')
    command = 'snmp-server group {group}'.format(group=group)
    if config_data.get('version'):
        command += ' {version}'.format(version=config_data['version'])
    if config_data.get('notify'):
        command += ' notify {notify}'.format(notify=config_data['notify'])
    if config_data.get('read'):
        command += ' read {read}'.format(read=config_data['read'])
    if config_data.get('write'):
        command += ' write {write}'.format(write=config_data['write'])
    if config_data.get('context'):
        command += ' context {context}'.format(context=config_data['context'])
    if config_data.get('acl_v4'):
        command += ' IPv4 {acl_v4}'.format(acl_v4=config_data['acl_v4'])
    if config_data.get('acl_v6'):
        command += ' IPv6 {acl_v6}'.format(acl_v6=config_data['acl_v6'])
    if config_data.get('v4_acl'):
        command += ' {v4_acl}'.format(v4_acl=config_data['v4_acl'])
    return command