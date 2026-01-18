from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def add_lun_map_reporting_nodes(self, nodes):
    reporting_nodes_obj = netapp_utils.zapi.NaElement('lun-map-add-reporting-nodes')
    reporting_nodes_obj.add_new_child('igroup', self.parameters['initiator_group_name'])
    reporting_nodes_obj.add_new_child('path', self.parameters['path'])
    nodes_obj = netapp_utils.zapi.NaElement('nodes')
    for node in nodes:
        nodes_obj.add_new_child('filer-id', node)
    reporting_nodes_obj.add_child_elem(nodes_obj)
    try:
        self.server.invoke_successfully(reporting_nodes_obj, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error creating LUN map reporting nodes for %s: %s' % (self.parameters['initiator_group_name'], to_native(error)), exception=traceback.format_exc())