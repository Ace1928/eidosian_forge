from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
from datetime import datetime
def get_cg_obj(self, name):
    try:
        return self.unity_conn.get_cg(name=name)
    except Exception as e:
        error_msg = 'Failed to get cg %s with error %s' % (name, str(e))
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)