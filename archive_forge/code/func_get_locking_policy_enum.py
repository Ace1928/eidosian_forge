from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_locking_policy_enum(self, locking_policy):
    """Get the locking_policy enum.
             :param locking_policy: The locking_policy string
             :return: locking_policy enum
        """
    if locking_policy in utils.FSLockingPolicyEnum.__members__:
        return utils.FSLockingPolicyEnum[locking_policy]
    else:
        errormsg = 'Invalid choice {0} for locking_policy'.format(locking_policy)
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)