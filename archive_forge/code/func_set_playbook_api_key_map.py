from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def set_playbook_api_key_map(self):
    self.na_helper.params_to_rest_api_keys = {'job_minutes': 'minutes', 'job_months': 'months', 'job_hours': 'hours', 'job_days_of_month': 'days', 'job_days_of_week': 'weekdays'}