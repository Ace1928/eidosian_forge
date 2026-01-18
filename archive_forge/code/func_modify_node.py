from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
import copy
def modify_node(self, modify=None, uuid=None):
    """
        Modify an existing node
        :return: none
        """
    if self.use_rest:
        self.update_node_details(uuid, modify)
    else:
        node_obj = netapp_utils.zapi.NaElement('system-node-modify')
        node_obj.add_new_child('node', self.parameters['name'])
        if 'location' in self.parameters:
            node_obj.add_new_child('node-location', self.parameters['location'])
        if 'asset_tag' in self.parameters:
            node_obj.add_new_child('node-asset-tag', self.parameters['asset_tag'])
        try:
            self.cluster.invoke_successfully(node_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error modifying node: %s' % to_native(error), exception=traceback.format_exc())