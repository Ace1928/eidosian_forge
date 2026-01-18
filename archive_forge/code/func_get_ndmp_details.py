from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
def get_ndmp_details(self, ndmp_details, ndmp_attributes):
    """
        :param ndmp_details: a dict of current ndmp.
        :param ndmp_attributes: ndmp returned from api call in xml format.
        :return: None
        """
    for option in self.modifiable_options:
        option_type = self.modifiable_options[option]['type']
        if option_type == 'bool':
            ndmp_details[option] = self.str_to_bool(ndmp_attributes.get_child_content(self.attribute_to_name(option)))
        elif option_type == 'int':
            ndmp_details[option] = int(ndmp_attributes.get_child_content(self.attribute_to_name(option)))
        elif option_type == 'list':
            child_list = ndmp_attributes.get_child_by_name(self.attribute_to_name(option))
            values = [child.get_content() for child in child_list.get_children()]
            ndmp_details[option] = values
        else:
            ndmp_details[option] = ndmp_attributes.get_child_content(self.attribute_to_name(option))