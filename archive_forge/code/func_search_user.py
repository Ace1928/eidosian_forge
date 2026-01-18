from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
def search_user(self):
    """
        Search the specified user from ESXi

        Returns: searched user
        """
    searchStr = self.user_name
    exactMatch = True
    findUsers = True
    findGroups = False
    user_account = self.host_obj.configManager.userDirectory.RetrieveUserGroups(None, searchStr, None, None, exactMatch, findUsers, findGroups)
    return user_account