from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_job_schedule(self):
    """
        Return details about the job
        :param:
            name : Job name
        :return: Details about the Job. None if not found.
        :rtype: dict
        """
    if self.use_rest:
        return self.get_job_schedule_rest()
    job_get_iter = netapp_utils.zapi.NaElement('job-schedule-cron-get-iter')
    query = {'job-schedule-cron-info': {'job-schedule-name': self.parameters['name']}}
    if self.parameters.get('cluster'):
        query['job-schedule-cron-info']['job-schedule-cluster'] = self.parameters['cluster']
    job_get_iter.translate_struct({'query': query})
    try:
        result = self.server.invoke_successfully(job_get_iter, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching job schedule %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
    job_details = None
    if result.get_child_by_name('num-records') and int(result['num-records']) >= 1:
        job_info = result['attributes-list']['job-schedule-cron-info']
        job_details = {}
        for item_key, zapi_key in self.na_helper.zapi_string_keys.items():
            job_details[item_key] = job_info[zapi_key]
        for item_key, zapi_key in self.na_helper.zapi_list_keys.items():
            parent, dummy = zapi_key
            job_details[item_key] = self.na_helper.get_value_for_list(from_zapi=True, zapi_parent=job_info.get_child_by_name(parent))
            if item_key == 'job_months' and self.month_offset == 1:
                job_details[item_key] = [int(x) + 1 if int(x) >= 0 else int(x) for x in job_details[item_key]]
            elif item_key == 'job_minutes' and len(job_details[item_key]) == 60:
                job_details[item_key] = [-1]
            else:
                job_details[item_key] = [int(x) for x in job_details[item_key]]
            if not job_details[item_key]:
                job_details[item_key] = [-1]
    return job_details