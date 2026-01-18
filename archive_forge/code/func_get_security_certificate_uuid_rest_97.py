from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_security_certificate_uuid_rest_97(self, name, type):
    query = {'common_name': name, 'type': type}
    fields = 'uuid,common_name,type'
    return self._get_security_certificate_uuid_rest_any(query, fields)