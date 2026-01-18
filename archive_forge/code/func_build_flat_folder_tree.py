from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
def build_flat_folder_tree(self, folder, path):
    ret = []
    tree = {'path': path, 'moid': folder._moId}
    ret.append(tree)
    children = None
    if hasattr(folder, 'childEntity'):
        children = folder.childEntity
    if children:
        for child in children:
            if child == folder:
                continue
            if isinstance(child, vim.Folder):
                ret.extend(self.build_flat_folder_tree(child, '%s/%s' % (path, child.name)))
    return ret