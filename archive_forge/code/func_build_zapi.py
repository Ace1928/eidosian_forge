from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def build_zapi(self, name):
    """ build ZAPI request based on resource  name """
    if name == 'sp_upgrade':
        zapi_obj = netapp_utils.zapi.NaElement('service-processor-image-update-progress-get')
        zapi_obj.add_new_child('node', self.parameters['attributes']['node'])
        return zapi_obj
    if name == 'sp_version':
        zapi_obj = netapp_utils.zapi.NaElement('service-processor-get')
        zapi_obj.add_new_child('node', self.parameters['attributes']['node'])
        return zapi_obj
    if name in self.resource_configuration:
        self.module.fail_json(msg='Error: event %s is not supported with ZAPI.  It requires REST.' % name)
    raise KeyError(name)