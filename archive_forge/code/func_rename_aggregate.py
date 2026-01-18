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
def rename_aggregate(self):
    """
        Rename aggregate.
        """
    if self.use_rest:
        return self.rename_aggr_rest()
    aggr_rename = netapp_utils.zapi.NaElement.create_node_with_children('aggr-rename', **{'aggregate': self.parameters['from_name'], 'new-aggregate-name': self.parameters['name']})
    try:
        self.server.invoke_successfully(aggr_rename, enable_tunneling=False)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error renaming aggregate %s: %s' % (self.parameters['from_name'], to_native(error)), exception=traceback.format_exc())