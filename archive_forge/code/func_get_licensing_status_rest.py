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
def get_licensing_status_rest(self):
    api = 'cluster/licensing/licenses'
    query = {'state': 'compliant, noncompliant, unlicensed, unknown'}
    fields = 'name,state,licenses'
    records, error = rest_generic.get_0_or_more_records(self.rest_api, api, query, fields)
    if error:
        self.module.fail_json(msg=error)
    current = {'installed_licenses': {}}
    if records:
        for package in records:
            current[package['name']] = package['state']
            if 'licenses' in package:
                for license in package['licenses']:
                    installed_license = license.get('installed_license')
                    serial_number = license.get('serial_number')
                    if serial_number and installed_license:
                        if serial_number not in current:
                            current['installed_licenses'][serial_number] = set()
                        current['installed_licenses'][serial_number].add(installed_license)
    return (current, records)