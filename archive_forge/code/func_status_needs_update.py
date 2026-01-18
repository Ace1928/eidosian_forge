from __future__ import absolute_import, division, print_function
import json
from ansible_collections.community.network.plugins.module_utils.network.a10.a10 import (axapi_call, a10_argument_spec, axapi_authenticate, axapi_failure,
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import url_argument_spec
def status_needs_update(current_status, new_status):
    """
                Check to determine if we want to change the status of a server.
                If there is a difference between the current status of the server and
                the desired status, return true, otherwise false.
                """
    if current_status != new_status:
        return True
    return False