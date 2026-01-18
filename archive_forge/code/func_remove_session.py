from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.consul import (
def remove_session(module, consul_module):
    session_id = module.params.get('id')
    try:
        destroy_session(consul_module, session_id)
        module.exit_json(changed=True, session_id=session_id)
    except Exception as e:
        module.fail_json(msg="Could not remove session with id '%s' %s" % (session_id, e))