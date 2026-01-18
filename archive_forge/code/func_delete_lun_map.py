from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
import codecs
from ansible.module_utils._text import to_text, to_bytes
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_lun_map(self):
    """
        Unmap LUN map
        """
    if self.use_rest:
        return self.delete_lun_map_rest()
    lun_map_delete = netapp_utils.zapi.NaElement.create_node_with_children('lun-unmap', **{'path': self.parameters['path'], 'initiator-group': self.parameters['initiator_group_name']})
    try:
        self.server.invoke_successfully(lun_map_delete, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as e:
        self.module.fail_json(msg='Error unmapping lun %s of initiator_group_name %s: %s' % (self.parameters['path'], self.parameters['initiator_group_name'], to_native(e)), exception=traceback.format_exc())