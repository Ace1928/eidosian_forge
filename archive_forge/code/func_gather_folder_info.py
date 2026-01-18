from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
def gather_folder_info(self):
    datacenter = self.find_datacenter_by_name(self.dc_name)
    if datacenter is None:
        self.module.fail_json(msg='Failed to find the datacenter %s' % self.dc_name)
    folder_trees = {}
    folder_trees['vmFolders'] = self.build_folder_tree(datacenter.vmFolder, '/%s/vm' % self.dc_name)
    folder_trees['hostFolders'] = self.build_folder_tree(datacenter.hostFolder, '/%s/host' % self.dc_name)
    folder_trees['networkFolders'] = self.build_folder_tree(datacenter.networkFolder, '/%s/network' % self.dc_name)
    folder_trees['datastoreFolders'] = self.build_folder_tree(datacenter.datastoreFolder, '/%s/datastore' % self.dc_name)
    flat_folder_info = self.build_flat_folder_tree(datacenter.vmFolder, '/%s/vm' % self.dc_name)
    flat_folder_info.extend(self.build_flat_folder_tree(datacenter.hostFolder, '/%s/host' % self.dc_name))
    flat_folder_info.extend(self.build_flat_folder_tree(datacenter.networkFolder, '/%s/network' % self.dc_name))
    flat_folder_info.extend(self.build_flat_folder_tree(datacenter.datastoreFolder, '/%s/datastore' % self.dc_name))
    self.module.exit_json(changed=False, folder_info=folder_trees, flat_folder_info=flat_folder_info)