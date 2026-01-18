from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_pool_type_enum(self, pool_type):
    """ Get the storage pool_type enum.
             :param pool_type: The pool_type
             :return: pool_type enum
        """
    if pool_type == 'TRADITIONAL':
        return 1
    elif pool_type == 'DYNAMIC':
        return 2
    else:
        errormsg = 'Invalid choice %s for Storage Pool Type' % pool_type
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)