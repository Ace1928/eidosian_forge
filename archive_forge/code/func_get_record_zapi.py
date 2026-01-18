from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_record_zapi(self, name, zapi_obj):
    """ calls the ZAPI and extract condition value"""
    try:
        results = self.server.invoke_successfully(zapi_obj, True)
    except netapp_utils.zapi.NaApiError as error:
        return (None, 'Error running command %s: %s' % (self.parameters['name'], to_native(error)))
    return (results, None)