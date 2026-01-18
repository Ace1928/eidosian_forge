from __future__ import absolute_import, division, print_function
import json
from ansible_collections.community.network.plugins.module_utils.network.a10.a10 import (axapi_call, a10_argument_spec, axapi_authenticate, axapi_failure,
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import url_argument_spec
def port_needs_update(src_ports, dst_ports):
    """
                Checks to determine if the port definitions of the src_ports
                array are in or different from those in dst_ports. If there is
                a difference, this function returns true, otherwise false.
                """
    for src_port in src_ports:
        found = False
        different = False
        for dst_port in dst_ports:
            if src_port['port_num'] == dst_port['port_num']:
                found = True
                for valid_field in VALID_PORT_FIELDS:
                    if src_port[valid_field] != dst_port[valid_field]:
                        different = True
                        break
                if found or different:
                    break
        if not found or different:
            return True
    return False