from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@api_wrapper
def get_host_by_name(system, host_name):
    """Find a host by the name specified in the module"""
    host = None
    for a_host in system.hosts.to_list():
        a_host_name = a_host.get_name()
        if a_host_name == host_name:
            host = a_host
            break
    return host