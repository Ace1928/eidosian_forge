from __future__ import absolute_import, division, print_function
import logging
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def rename_cg(self, cg_name, new_cg_name):
    """Rename consistency group.
            :param cg_name: The name of the consistency group
            :param new_cg_name: The new name of the consistency group
            :return: Boolean value to indicate if consistency group renamed
        """
    cg_obj = self.return_cg_instance(cg_name)
    try:
        cg_obj.modify(name=new_cg_name)
        return True
    except Exception as e:
        errormsg = 'Rename operation of consistency group {0} failed with error {1}'.format(cg_name, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)