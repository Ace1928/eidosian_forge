from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_job_schedule_rest(self):
    """
        Return details about the job
        :param:
            name : Job name
        :return: Details about the Job. None if not found.
        :rtype: dict
        """
    query = {'name': self.parameters['name']}
    if self.parameters.get('cluster'):
        query['cluster'] = self.parameters['cluster']
    record, error = rest_generic.get_one_record(self.rest_api, 'cluster/schedules', query, 'uuid,cron')
    if error is not None:
        self.module.fail_json(msg='Error fetching job schedule: %s' % error)
    if record:
        self.uuid = record['uuid']
        job_details = {'name': record['name']}
        for param_key, rest_key in self.na_helper.params_to_rest_api_keys.items():
            if rest_key in record['cron']:
                job_details[param_key] = record['cron'][rest_key]
            else:
                job_details[param_key] = [-1]
        if 'job_months' in job_details and self.month_offset == 0:
            job_details['job_months'] = [x - 1 if x > 0 else x for x in job_details['job_months']]
        if 'job_minutes' in job_details and len(job_details['job_minutes']) == 60:
            job_details['job_minutes'] = [-1]
        return job_details
    return None