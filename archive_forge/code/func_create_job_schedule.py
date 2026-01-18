from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_job_schedule(self):
    """
        Creates a job schedule
        """
    if self.use_rest:
        cron = {}
        for param_key, rest_key in self.na_helper.params_to_rest_api_keys.items():
            if self.parameters.get(param_key):
                if len(self.parameters[param_key]) == 1 and self.parameters[param_key][0] == -1:
                    if rest_key == 'minutes':
                        cron[rest_key] = []
                elif param_key == 'job_months' and self.month_offset == 0:
                    cron[rest_key] = [x + 1 if x >= 0 else x for x in self.parameters[param_key]]
                else:
                    cron[rest_key] = self.parameters[param_key]
        params = {'name': self.parameters['name'], 'cron': cron}
        if self.parameters.get('cluster'):
            params['cluster'] = self.parameters['cluster']
        api = 'cluster/schedules'
        dummy, error = self.rest_api.post(api, params)
        if error is not None:
            self.module.fail_json(msg='Error creating job schedule: %s' % error)
    else:
        job_schedule_create = netapp_utils.zapi.NaElement('job-schedule-cron-create')
        self.add_job_details(job_schedule_create, self.parameters)
        try:
            self.server.invoke_successfully(job_schedule_create, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error creating job schedule %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())