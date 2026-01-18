from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def patch_aggr_rest(self, action, body, query=None):
    api = 'storage/aggregates'
    dummy, error = rest_generic.patch_async(self.rest_api, api, self.uuid, body, query)
    if error:
        self.module.fail_json(msg='Error: failed to %s aggregate: %s' % (action, error))