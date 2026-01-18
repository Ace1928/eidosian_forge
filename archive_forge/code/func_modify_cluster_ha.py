from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_cluster_ha(self, configure):
    """
        Enable or disable HA on cluster
        :return: None
        """
    if self.use_rest:
        return self.modify_cluster_ha_rest(configure)
    cluster_ha_modify = netapp_utils.zapi.NaElement.create_node_with_children('cluster-ha-modify', **{'ha-configured': configure})
    try:
        self.server.invoke_successfully(cluster_ha_modify, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error modifying cluster HA to %s: %s' % (configure, to_native(error)), exception=traceback.format_exc())