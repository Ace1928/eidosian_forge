from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_security_ipsec_config(self):
    """Get IPsec config details"""
    record, error = rest_generic.get_one_record(self.rest_api, 'security/ipsec', None, 'enabled,replay_window')
    if error:
        self.module.fail_json(msg='Error fetching security IPsec config: %s' % to_native(error), exception=traceback.format_exc())
    if record:
        return {'enabled': record.get('enabled'), 'replay_window': record.get('replay_window')}
    return None