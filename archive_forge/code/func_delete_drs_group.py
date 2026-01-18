from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def delete_drs_group(self):
    """
        Function to delete a DRS host/vm group
        """
    if self.__group_obj is not None:
        self.__changed = True
        if not self.module.check_mode:
            group_spec = vim.cluster.GroupSpec(removeKey=self.__group_name, operation=self.__operation)
            config_spec = vim.cluster.ConfigSpecEx(groupSpec=[group_spec])
            task = self.__cluster_obj.ReconfigureEx(config_spec, modify=True)
            wait_for_task(task)
    if self.__changed:
        self.__msg = 'Deleted group `%s` successfully' % self.__group_name
    else:
        self.__msg = 'DRS group `%s` does not exists or already deleted' % self.__group_name