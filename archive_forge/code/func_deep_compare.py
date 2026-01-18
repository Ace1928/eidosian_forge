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
def deep_compare(self, records):
    """ look for any change in license details, capacity, expiration, ...
            this is run after apply, so we don't know for sure in check_mode
        """
    if not HAS_DEEPDIFF:
        self.module.warn('deepdiff is required to identify detailed changes')
        return []
    diffs = DeepDiff(self.previous_records, records)
    self.rest_api.log_debug('diffs', diffs)
    roots = set(re.findall('root\\[(\\d+)\\]', str(diffs)))
    result = [records[int(index)]['name'] for index in roots]
    self.rest_api.log_debug('deep_changed_keys', result)
    return result