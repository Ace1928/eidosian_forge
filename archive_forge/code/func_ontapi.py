from __future__ import absolute_import, division, print_function
import codecs
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_bytes, to_native, to_text
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def ontapi(self):
    """Method to get ontapi version"""
    api = 'system-get-ontapi-version'
    api_call = netapp_utils.zapi.NaElement(api)
    try:
        results = self.server.invoke_successfully(api_call, enable_tunneling=True)
        ontapi_version = results.get_child_content('minor-version')
        return ontapi_version if ontapi_version is not None else '0'
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error calling API %s: %s' % (api, to_native(error)), exception=traceback.format_exc())