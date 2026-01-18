from __future__ import absolute_import, division, print_function
import time
import traceback
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def quota_entry_modify(self, modify_attrs):
    """
        Modifies a quota entry
        """
    for key in list(modify_attrs):
        modify_attrs[key.replace('_', '-')] = modify_attrs.pop(key)
    options = {'volume': self.parameters['volume'], 'quota-target': self.parameters['quota_target'], 'quota-type': self.parameters['type'], 'qtree': self.parameters['qtree']}
    options.update(modify_attrs)
    self.set_zapi_options(options)
    if self.parameters.get('policy'):
        options['policy'] = str(self.parameters['policy'])
    modify_entry = netapp_utils.zapi.NaElement.create_node_with_children('quota-modify-entry', **options)
    try:
        self.server.invoke_successfully(modify_entry, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error modifying quota entry %s: %s' % (self.parameters['volume'], to_native(error)), exception=traceback.format_exc())