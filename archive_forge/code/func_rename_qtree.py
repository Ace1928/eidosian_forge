from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def rename_qtree(self):
    """
        Rename a qtree
        """
    if self.use_rest:
        error = 'Internal error, use modify with REST'
        self.module.fail_json(msg=error)
    else:
        path = '/vol/%s/%s' % (self.parameters['flexvol_name'], self.parameters['from_name'])
        new_path = '/vol/%s/%s' % (self.parameters['flexvol_name'], self.parameters['name'])
        qtree_rename = netapp_utils.zapi.NaElement.create_node_with_children('qtree-rename', **{'qtree': path, 'new-qtree-name': new_path})
        try:
            self.server.invoke_successfully(qtree_rename, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error renaming qtree %s: %s' % (self.parameters['from_name'], to_native(error)), exception=traceback.format_exc())