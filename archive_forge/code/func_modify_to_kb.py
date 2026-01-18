from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_to_kb(self, converted_parameters):
    """
        Save a coverted parameter
        :param converted_parameters: Dic of all parameters
        :return:
        """
    for attr in ['maximum_size', 'minimum_size', 'increment_size']:
        if converted_parameters.get(attr) is not None:
            if self.use_rest:
                converted_parameters[attr] = self.convert_to_byte(attr, converted_parameters)
            else:
                converted_parameters[attr] = str(self.convert_to_kb(attr, converted_parameters))
    return converted_parameters