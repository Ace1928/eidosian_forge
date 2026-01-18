from __future__ import absolute_import, division, print_function
import re
import socket
import sys
import traceback
from ansible.module_utils.basic import env_fallback
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list, ComplexList
from ansible.module_utils.connection import exec_command, ConnectionError
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import NetconfConnection
def cli_err_msg(cmd, err):
    """ get cli exception message"""
    if not err:
        return 'Error: Fail to get cli exception message.'
    msg = list()
    err_list = str(err).split('\r\n')
    for err in err_list:
        err = err.strip('.,\r\n\t ')
        if not err:
            continue
        if cmd and cmd == err:
            continue
        if " at '^' position" in err:
            err = err.replace(" at '^' position", '').strip()
        err = err.strip('.,\r\n\t ')
        if err == '^':
            continue
        if len(err) > 2 and err[0] in ['<', '['] and (err[-1] in ['>', ']']):
            continue
        err.strip('.,\r\n\t ')
        if err:
            msg.append(err)
    if cmd:
        msg.insert(0, 'Command: %s' % cmd)
    return ', '.join(msg).capitalize() + '.'