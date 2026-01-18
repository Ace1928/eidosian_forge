from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def compare_command(commands):
    """check if the commands is valid"""
    if len(commands) < 3:
        return True
    if commands[0] != 'sys' or commands[1] != 'undo ip route-static default-bfd' or commands[2] != 'commit':
        return True