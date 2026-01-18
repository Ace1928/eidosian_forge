from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def cluster_image_update_progress_get(self, ignore_connection_error=True):
    """
        Get current cluster image update progress info
        :return: Dictionary of cluster image update progress if query successful, else return None
        """
    cluster_update_progress_get = netapp_utils.zapi.NaElement('cluster-image-update-progress-info')
    cluster_update_progress_info = {}
    try:
        result = self.server.invoke_successfully(cluster_update_progress_get, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        if ignore_connection_error:
            return cluster_update_progress_info
        self.module.fail_json(msg='Error fetching cluster image update progress details: %s' % to_native(error), exception=traceback.format_exc())
    if result.get_child_by_name('attributes').get_child_by_name('ndu-progress-info'):
        update_progress_info = result.get_child_by_name('attributes').get_child_by_name('ndu-progress-info')
        cluster_update_progress_info['overall_status'] = update_progress_info.get_child_content('overall-status')
        cluster_update_progress_info['completed_node_count'] = update_progress_info.get_child_content('completed-node-count')
        reports = update_progress_info.get_child_by_name('validation-reports')
        if reports:
            cluster_update_progress_info['validation_reports'] = []
            for report in reports.get_children():
                checks = {}
                for check in report.get_children():
                    checks[self.get_localname(check.get_name())] = check.get_content()
                cluster_update_progress_info['validation_reports'].append(checks)
    return cluster_update_progress_info