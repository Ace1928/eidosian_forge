from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_modify(self, current):
    modify = self.na_helper.get_modified_attributes(current, self.parameters)
    if not modify or ('permission' in modify and len(modify) == 1):
        return modify
    if 'type' in modify:
        self.module.fail_json(msg='Error: changing the type is not supported by ONTAP - current: %s, desired: %s' % (current['type'], self.parameters['type']))
    self.module.fail_json(msg='Error: only permission can be changed - modify: %s' % modify)