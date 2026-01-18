from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_cluster_identity(self, modify):
    """
        Modifies the cluster identity
        """
    if self.use_rest:
        return self.modify_cluster_identity_rest(modify)
    cluster_modify = netapp_utils.zapi.NaElement('cluster-identity-modify')
    if modify.get('cluster_name') is not None:
        cluster_modify.add_new_child('cluster-name', modify.get('cluster_name'))
    if modify.get('cluster_location') is not None:
        cluster_modify.add_new_child('cluster-location', modify.get('cluster_location'))
    if modify.get('cluster_contact') is not None:
        cluster_modify.add_new_child('cluster-contact', modify.get('cluster_contact'))
    try:
        self.server.invoke_successfully(cluster_modify, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error modifying cluster idetity details %s: %s' % (self.parameters['cluster_name'], to_native(error)), exception=traceback.format_exc())