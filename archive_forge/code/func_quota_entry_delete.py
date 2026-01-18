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
def quota_entry_delete(self):
    """
        Deletes a quota entry
        """
    options = {'volume': self.parameters['volume'], 'quota-target': self.parameters['quota_target'], 'quota-type': self.parameters['type'], 'qtree': self.parameters['qtree']}
    set_entry = netapp_utils.zapi.NaElement.create_node_with_children('quota-delete-entry', **options)
    if self.parameters.get('policy'):
        set_entry.add_new_child('policy', self.parameters['policy'])
    try:
        self.server.invoke_successfully(set_entry, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error deleting quota entry %s: %s' % (self.parameters['volume'], to_native(error)), exception=traceback.format_exc())