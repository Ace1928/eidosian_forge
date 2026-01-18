from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_access_policy_enum(self, access_policy):
    """Get the access_policy enum.
             :param access_policy: The access_policy string
             :return: access_policy enum
        """
    if access_policy in utils.AccessPolicyEnum.__members__:
        return utils.AccessPolicyEnum[access_policy]
    else:
        errormsg = 'Invalid choice {0} for access_policy'.format(access_policy)
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)