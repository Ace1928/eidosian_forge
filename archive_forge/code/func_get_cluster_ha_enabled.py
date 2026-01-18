from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_cluster_ha_enabled(self):
    """
        Get current cluster HA details
        :return: dict if enabled, None if disabled
        """
    if self.use_rest:
        return self.get_cluster_ha_enabled_rest()
    cluster_ha_get = netapp_utils.zapi.NaElement('cluster-ha-get')
    try:
        result = self.server.invoke_successfully(cluster_ha_get, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError:
        self.module.fail_json(msg='Error fetching cluster HA details', exception=traceback.format_exc())
    cluster_ha_info = result.get_child_by_name('attributes').get_child_by_name('cluster-ha-info')
    if cluster_ha_info.get_child_content('ha-configured') == 'true':
        return {'ha-configured': True}
    return None