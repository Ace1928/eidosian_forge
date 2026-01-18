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
def get_delete_actions(self):
    packages_to_delete = []
    if self.parameters.get('license_names') is not None:
        for package in list(self.parameters['license_names']):
            if 'installed_licenses' in self.license_status and self.parameters['serial_number'] != '*' and (self.parameters['serial_number'] in self.license_status['installed_licenses']) and (package in self.license_status['installed_licenses'][self.parameters['serial_number']]):
                packages_to_delete.append(package)
            if package in self.license_status:
                packages_to_delete.append(package)
    for dummy, nlf_dict in self.nlfs:
        if nlf_dict:
            self.validate_delete_action(nlf_dict)
    nlfs_to_delete = [nlf_dict for dummy, nlf_dict in self.nlfs if self.nlf_is_installed(nlf_dict)]
    return (bool(nlfs_to_delete) or bool(self.parameters.get('license_names')), packages_to_delete, nlfs_to_delete)