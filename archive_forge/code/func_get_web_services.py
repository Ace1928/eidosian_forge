from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_web_services(self):
    record, error = rest_generic.get_one_record(self.rest_api, 'cluster/web', fields='certificate')
    if error:
        self.module.fail_json(msg='Error fetching cluster web service config: %s' % to_native(error), exception=traceback.format_exc())
    if record:
        return record
    return None