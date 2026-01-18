from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_volume
def check_job_status(self, jobid):
    """
        Loop until job is complete
        """
    server = self.server
    sleep_time = 5
    time_out = self.parameters['time_out']
    while time_out > 0:
        results = self.get_job(jobid, server)
        if results is None and server == self.server:
            results = netapp_utils.get_cserver(self.server)
            server = netapp_utils.setup_na_ontap_zapi(module=self.module, vserver=results)
            continue
        if results is None:
            error = 'cannot locate job with id: %s' % jobid
            break
        if results['job-state'] in ('queued', 'running'):
            time.sleep(sleep_time)
            time_out -= sleep_time
            continue
        if results['job-state'] in ('success', 'failure'):
            break
        else:
            self.module.fail_json(msg='Unexpected job status in: %s' % repr(results))
    if results is not None:
        if results['job-state'] == 'success':
            error = None
        elif results['job-state'] in ('queued', 'running'):
            error = 'job completion exceeded expected timer of: %s seconds' % self.parameters['time_out']
        elif results['job-completion'] is not None:
            error = results['job-completion']
        else:
            error = results['job-progress']
    return error