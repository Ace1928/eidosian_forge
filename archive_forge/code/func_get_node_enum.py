from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_node_enum(self, sp):
    """Get the storage processor enum.
             :param sp: The storage processor string
             :return: storage processor enum
        """
    if sp in utils.NodeEnum.__members__:
        return utils.NodeEnum[sp]
    else:
        errormsg = 'Invalid choice {0} for storage processor'.format(sp)
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)