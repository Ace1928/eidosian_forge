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
def nlf_is_installed(self, nlf_dict):
    """ return True if NLF with same SN, product (bundle) name and package list is present
            return False otherwise
            Even when present, the NLF may not be active, so this is only useful for delete
        """
    n_serial_number, n_product = self.get_sn_and_product(nlf_dict)
    if not n_product or not n_serial_number:
        return False
    if 'installed_licenses' not in self.license_status:
        return False
    if n_serial_number == '*' and self.parameters['state'] == 'absent':
        return True
    if n_serial_number not in self.license_status['installed_licenses']:
        return False
    return n_product in self.license_status['installed_licenses'][n_serial_number]