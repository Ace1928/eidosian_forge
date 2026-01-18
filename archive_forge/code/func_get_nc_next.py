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
def get_nc_next(module, xml_str):
    """ get_nc_next for exchange capability """
    conn = get_nc_connection(module)
    result = None
    if xml_str is not None:
        response = conn.get(xml_str, if_rpc_reply=True)
        result = response.find('./*')
        set_id = response.get('set-id')
        while set_id is not None:
            try:
                fetch_node = new_ele_ns('get-next', 'http://www.huawei.com/netconf/capability/base/1.0', {'set-id': set_id})
                next_xml = conn.dispatch_rpc(etree.tostring(fetch_node))
                if next_xml is not None:
                    result.extend(next_xml.find('./*'))
                set_id = next_xml.get('set-id')
            except ConnectionError:
                break
    if result is not None:
        return to_string(to_xml(result))
    return result