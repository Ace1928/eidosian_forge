from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def check_invoke_result(self, result, action):
    """
        check invoked api call back result.
        """
    results = {}
    for key in ('result-status', 'result-jobid'):
        if result.get_child_by_name(key):
            results[key] = result[key]
    status = results.get('result-status')
    if status == 'in_progress' and 'result-jobid' in results:
        if self.parameters['time_out'] == 0:
            return
        error = self.check_job_status(results['result-jobid'])
        if error is None:
            return
        else:
            self.wrap_fail_json(msg='Error when %s volume: %s' % (action, error))
    if status == 'failed':
        self.wrap_fail_json(msg='Operation failed when %s volume.' % action)