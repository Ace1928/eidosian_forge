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
def remove_licenses(self, package_name, nlf_dict=None):
    """
        Remove requested licenses
        :param:
          package_name: Name of the license to be deleted
        """
    if self.use_rest:
        return self.remove_licenses_rest(package_name, nlf_dict or {})
    license_delete = netapp_utils.zapi.NaElement('license-v2-delete')
    license_delete.add_new_child('serial-number', self.parameters['serial_number'])
    license_delete.add_new_child('package', package_name)
    try:
        self.server.invoke_successfully(license_delete, enable_tunneling=False)
        return True
    except netapp_utils.zapi.NaApiError as error:
        if to_native(error.code) == '15661':
            return False
        else:
            self.module.fail_json(msg='Error removing license %s' % to_native(error), exception=traceback.format_exc())