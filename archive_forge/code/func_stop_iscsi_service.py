from __future__ import absolute_import, division, print_function
import traceback
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def stop_iscsi_service(self):
    """
         Stop iscsi service
        """
    iscsi_stop = netapp_utils.zapi.NaElement.create_node_with_children('iscsi-service-stop')
    try:
        self.server.invoke_successfully(iscsi_stop, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as e:
        self.module.fail_json(msg='Error Stopping iscsi service on vserver %s: %s' % (self.parameters['vserver'], to_native(e)), exception=traceback.format_exc())