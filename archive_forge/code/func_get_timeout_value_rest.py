from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_timeout_value_rest(self):
    """ Get CLI inactivity timeout value """
    fields = 'timeout'
    api = 'private/cli/system/timeout'
    record, error = rest_generic.get_one_record(self.rest_api, api, query=None, fields=fields)
    if error:
        self.module.fail_json(msg='Error fetching CLI sessions timeout value: %s' % to_native(error), exception=traceback.format_exc())
    if record:
        return {'timeout': record.get('timeout')}
    return None