from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def search_folder(self, folder_name):
    """
            Search folder in vCenter
            Returns: folder object
        """
    search_index = self.content.searchIndex
    folder_obj = search_index.FindByInventoryPath(folder_name)
    if not (folder_obj and isinstance(folder_obj, vim.Folder)):
        self.module.fail_json(msg="Folder '%s' not found" % folder_name)
    return folder_obj