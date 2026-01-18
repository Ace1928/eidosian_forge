from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def is_fabricpool(self, name, aggregate_uuid):
    """whether the aggregate is associated with one or more object stores"""
    api = 'storage/aggregates/%s/cloud-stores' % aggregate_uuid
    records, error = rest_generic.get_0_or_more_records(self.rest_api, api)
    if error:
        self.module.fail_json(msg='Error getting object store for aggregate: %s: %s' % (name, error))
    return records is not None and len(records) > 0