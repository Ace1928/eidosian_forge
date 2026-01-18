from __future__ import (absolute_import, division, print_function)
import re
import os
import sys
import time
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config
from ansible.module_utils.connection import ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import validate_ip_v6_address
def get_cli_exception(exc=None):
    """Get cli exception message"""
    msg = list()
    if not exc:
        exc = sys.exc_info[1]
    if exc:
        errs = str(exc).split('\r\n')
        for err in errs:
            if not err:
                continue
            if 'matched error in response:' in err:
                continue
            if " at '^' position" in err:
                err = err.replace(" at '^' position", '')
            if err.replace(' ', '') == '^':
                continue
            if len(err) > 2 and err[0] in ['<', '['] and (err[-1] in ['>', ']']):
                continue
            if err[-1] == '.':
                err = err[:-1]
            if err.replace(' ', '') == '':
                continue
            msg.append(err)
    else:
        msg = ['Error: Fail to get cli exception message.']
    while msg[-1][-1] == ' ':
        msg[-1] = msg[-1][:-1]
    if msg[-1][-1] != '.':
        msg[-1] += '.'
    return ', '.join(msg).capitalize()