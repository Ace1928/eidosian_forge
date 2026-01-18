from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_modify_attributes(self, current, split):
    """
        :param current: current state.
        :param split: True or False of split action.
        :return: list of modified attributes.
        """
    modify = None
    if self.parameters['state'] == 'present':
        if self.parameters.get('from_name'):
            if split:
                modify = self.na_helper.get_modified_attributes(current, self.parameters)
                if modify.get('ports'):
                    del modify['ports']
        else:
            modify = self.na_helper.get_modified_attributes(current, self.parameters)
    return modify