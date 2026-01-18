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
def create_aggr_rest(self):
    api = 'storage/aggregates'
    disk_size = self.get_disk_size()
    query = {'return_records': 'true'}
    if disk_size:
        query['disk_size'] = disk_size
    body = {'name': self.parameters['name']} if 'name' in self.parameters else {}
    block_storage = {}
    primary = {}
    if self.parameters.get('nodes'):
        body['node.name'] = self.parameters['nodes'][0]
    if self.parameters.get('disk_class'):
        primary['disk_class'] = self.parameters['disk_class']
    if self.parameters.get('disk_count'):
        primary['disk_count'] = self.parameters['disk_count']
    if self.parameters.get('raid_size'):
        primary['raid_size'] = self.parameters['raid_size']
    if self.parameters.get('raid_type'):
        primary['raid_type'] = self.parameters['raid_type']
    if primary:
        block_storage['primary'] = primary
    mirror = {}
    if self.parameters.get('is_mirrored'):
        mirror['enabled'] = self.parameters['is_mirrored']
    if mirror:
        block_storage['mirror'] = mirror
    if block_storage:
        body['block_storage'] = block_storage
    if self.parameters.get('encryption'):
        body['data_encryption'] = {'software_encryption_enabled': True}
    if self.parameters.get('snaplock_type'):
        body['snaplock_type'] = self.parameters['snaplock_type']
    if self.parameters.get('tags') is not None:
        body['_tags'] = self.parameters['tags']
    response, error = rest_generic.post_async(self.rest_api, api, body or None, query, job_timeout=self.parameters['time_out'])
    if error:
        self.module.fail_json(msg='Error: failed to create aggregate: %s' % error)
    if response:
        record, error = rrh.check_for_0_or_1_records(api, response, error, query)
        if not error and record and ('uuid' not in record):
            error = 'uuid key not present in %s:' % record
        if error:
            self.module.fail_json(msg='Error: failed to parse create aggregate response: %s' % error)
        if record:
            self.uuid = record['uuid']