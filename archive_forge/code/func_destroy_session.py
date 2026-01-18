from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.consul import (
def destroy_session(consul_module, session_id):
    return consul_module.put(('session', 'destroy', session_id))