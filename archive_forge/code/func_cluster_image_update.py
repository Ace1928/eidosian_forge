from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def cluster_image_update(self):
    """
        Update current cluster image
        """
    cluster_update_info = netapp_utils.zapi.NaElement('cluster-image-update')
    cluster_update_info.add_new_child('package-version', self.parameters['package_version'])
    cluster_update_info.add_new_child('ignore-validation-warning', str(self.parameters['ignore_validation_warning']))
    if self.parameters.get('stabilize_minutes'):
        cluster_update_info.add_new_child('stabilize-minutes', self.na_helper.get_value_for_int(False, self.parameters['stabilize_minutes']))
    if self.parameters.get('nodes'):
        cluster_nodes = netapp_utils.zapi.NaElement('nodes')
        for node in self.parameters['nodes']:
            cluster_nodes.add_new_child('node-name', node)
        cluster_update_info.add_child_elem(cluster_nodes)
    try:
        self.server.invoke_successfully(cluster_update_info, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        msg = 'Error updating cluster image for %s: %s' % (self.parameters['package_version'], to_native(error))
        cluster_update_progress_info = self.cluster_image_update_progress_get(ignore_connection_error=True)
        validation_reports = cluster_update_progress_info.get('validation_reports')
        if validation_reports is None:
            validation_reports = self.cluster_image_validate()
        self.module.fail_json(msg=msg, validation_reports=str(validation_reports), validation_reports_after_download=self.validation_reports_after_download, validation_reports_after_update=validation_reports, exception=traceback.format_exc())