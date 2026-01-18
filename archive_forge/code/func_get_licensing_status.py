from __future__ import absolute_import, division, print_function
import re
import sys
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_licensing_status(self):
    """
            Check licensing status

            :return: package (key) and licensing status (value)
            :rtype: dict
        """
    if self.use_rest:
        return self.get_licensing_status_rest()
    license_status = netapp_utils.zapi.NaElement('license-v2-status-list-info')
    result = None
    try:
        result = self.server.invoke_successfully(license_status, enable_tunneling=False)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error checking license status: %s' % to_native(error), exception=traceback.format_exc())
    return_dictionary = {}
    license_v2_status = result.get_child_by_name('license-v2-status')
    if license_v2_status:
        for license_v2_status_info in license_v2_status.get_children():
            package = license_v2_status_info.get_child_content('package')
            status = license_v2_status_info.get_child_content('method')
            return_dictionary[package] = status
    return (return_dictionary, None)