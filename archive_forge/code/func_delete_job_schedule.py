from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_job_schedule(self):
    """
        Delete a job schedule
        """
    if self.use_rest:
        api = 'cluster/schedules/' + self.uuid
        dummy, error = self.rest_api.delete(api)
        if error is not None:
            self.module.fail_json(msg='Error deleting job schedule: %s' % error)
    else:
        job_schedule_delete = netapp_utils.zapi.NaElement('job-schedule-cron-destroy')
        self.add_job_details(job_schedule_delete, self.parameters)
        try:
            self.server.invoke_successfully(job_schedule_delete, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error deleting job schedule %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())