from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def create_volume_attribute(self, zapi_object, parent_attribute, attribute, option_name, convert_from=None):
    """

        :param parent_attribute:
        :param child_attribute:
        :param value:
        :return:
        """
    value = self.parameters.get(option_name)
    if value is None:
        return
    if convert_from == int:
        value = str(value)
    elif convert_from == bool:
        value = self.na_helper.get_value_for_bool(False, value, option_name)
    if zapi_object is None:
        parent_attribute.add_new_child(attribute, value)
        return
    if isinstance(zapi_object, str):
        element = parent_attribute.get_child_by_name(zapi_object)
        zapi_object = netapp_utils.zapi.NaElement(zapi_object) if element is None else element
    zapi_object.add_new_child(attribute, value)
    parent_attribute.add_child_elem(zapi_object)