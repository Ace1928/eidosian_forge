from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def rename_igroup(self):
    """
        Rename the igroup.
        """
    if self.use_rest:
        self.module.fail_json(msg='Internal error, should not call rename, but use modify')
    igroup_rename = netapp_utils.zapi.NaElement.create_node_with_children('igroup-rename', **{'initiator-group-name': self.parameters['from_name'], 'initiator-group-new-name': str(self.parameters['name'])})
    try:
        self.server.invoke_successfully(igroup_rename, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error renaming igroup %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())