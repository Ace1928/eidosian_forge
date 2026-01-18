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
def attach_object_store_to_aggr_rest(self):
    """TODO: support mirror in addition to primary"""
    if self.uuid is None:
        error = 'aggregate UUID is not set.'
        self.module.fail_json(msg='Error: cannot attach cloud store with name %s: %s' % (self.parameters['object_store_name'], error))
    body = {'target': {'uuid': self.get_cloud_target_uuid_rest()}}
    api = 'storage/aggregates/%s/cloud-stores' % self.uuid
    query = None
    if 'allow_flexgroups' in self.parameters:
        query = {'allow_flexgroups': 'true' if self.parameters['allow_flexgroups'] else 'false'}
    record, error = rest_generic.post_async(self.rest_api, api, body, query)
    if error:
        self.module.fail_json(msg='Error: failed to attach cloud store with name %s: %s' % (self.parameters['object_store_name'], error))
    return record