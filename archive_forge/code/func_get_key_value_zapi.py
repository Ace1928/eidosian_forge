from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_key_value_zapi(self, xml, key):
    for child in xml.get_children():
        value = xml.get_child_content(key)
        if value is not None:
            return value
        value = self.get_key_value(child, key)
        if value is not None:
            return value
    return None