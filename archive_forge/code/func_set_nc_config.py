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
def set_nc_config(module, xml_str):
    """ set_config """
    conn = get_nc_connection(module)
    try:
        out = conn.edit_config(target='running', config=xml_str, default_operation='merge', error_option='rollback-on-error')
    except Exception as e:
        message = re.findall('<error-message xml:lang=\\"en\\">(.*)</error-message>', str(e))
        if message:
            module.fail_json(msg='Error: %s' % message[0])
        else:
            module.fail_json(msg='Error: %s' % str(e))
    else:
        return to_string(to_xml(out))