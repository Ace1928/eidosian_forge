from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, find_obj
def remove_custom_attributes(self):
    changed = False
    for ca in self.ca_list:
        for x in self.obj.customValue:
            if ca['key'] == x.key and x.value != '':
                changed = True
                if not self.module.check_mode:
                    self.content.customFieldsManager.SetField(entity=self.obj, key=ca['key'], value='')
    return {'changed': changed, 'failed': False}