from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_bytes
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.argspec.logging_global.logging_global import (
def parse_console_node(self, conf, console_dict=None):
    console_loggings = ['any', 'authorization', 'change-log', 'conflict-log', 'daemon', 'dfc', 'external', 'firewall', 'ftp', 'interactive-commands', 'kernel', 'ntp', 'pfe', 'security', 'user']
    if console_dict is None:
        console_dict = {}
    if isinstance(conf, dict):
        for item in console_loggings:
            if item in conf.get('name'):
                any_dict = {}
                for k, v in conf.items():
                    if k != 'name':
                        any_dict['level'] = k
                console_dict[item.replace('-', '_')] = any_dict
    else:
        for console in conf:
            for item in console_loggings:
                if item in console.get('name'):
                    any_dict = {}
                    for k, v in console.items():
                        if k != 'name':
                            any_dict['level'] = k
                    console_dict[item.replace('-', '_')] = any_dict
    return console_dict