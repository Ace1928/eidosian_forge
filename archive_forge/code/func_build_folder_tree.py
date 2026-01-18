from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
def build_folder_tree(self, folder, path):
    tree = {'path': path, 'subfolders': {}, 'moid': folder._moId}
    children = None
    if hasattr(folder, 'childEntity'):
        children = folder.childEntity
    if children:
        for child in children:
            if child == folder:
                continue
            if isinstance(child, vim.Folder):
                ctree = self.build_folder_tree(child, '%s/%s' % (path, child.name))
                tree['subfolders'][child.name] = dict.copy(ctree)
    return tree