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
def add_licenses_rest(self):
    api = 'cluster/licensing/licenses'
    body = {'keys': [x[0] for x in self.nlfs]}
    headers = None
    if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 9, 1):
        headers = {'X-Dot-Error-Arguments': 'true'}
    dummy, error = rest_generic.post_async(self.rest_api, api, body, headers=headers)
    if error:
        error = self.format_post_error(error, body)
        if 'conflicts' in error:
            return error
        self.module.fail_json(msg='Error adding license: %s - previous license status: %s' % (error, self.license_status))
    return None