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
def get_object_store_action(self, current, rename):
    object_store_cd_action = None
    if self.parameters.get('object_store_name'):
        aggr_name = self.parameters['from_name'] if rename else self.parameters['name']
        object_store_current = self.get_object_store(aggr_name) if current else None
        object_store_cd_action = self.na_helper.get_cd_action(object_store_current, self.parameters.get('object_store_name'))
        if object_store_cd_action is None and object_store_current is not None and (object_store_current['object_store_name'] != self.parameters.get('object_store_name')):
            self.module.fail_json(msg='Error: object store %s is already associated with aggregate %s.' % (object_store_current['object_store_name'], aggr_name))
    return object_store_cd_action