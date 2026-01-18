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
def compare_license_status(self, previous_license_status):
    changed_keys = []
    for __ in range(5):
        error = None
        new_license_status, records = self.get_licensing_status()
        try:
            changed_keys = local_cmp(previous_license_status, new_license_status)
            break
        except KeyError as exc:
            error = exc
            time.sleep(5)
    if error:
        self.module.fail_json(msg='Error: mismatch in license package names: %s.  Expected: %s, found: %s.' % (error, previous_license_status.keys(), new_license_status.keys()))
    if 'installed_licenses' in changed_keys:
        changed_keys.remove('installed_licenses')
    if records and self.previous_records:
        deep_changed_keys = self.deep_compare(records)
        for key in deep_changed_keys:
            if key not in changed_keys:
                changed_keys.append(key)
    return changed_keys