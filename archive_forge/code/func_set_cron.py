from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def set_cron(param_key, rest_key, params, cron):
    if params[param_key] == [-1]:
        cron[rest_key] = []
    elif param_key == 'job_months' and self.month_offset == 0:
        cron[rest_key] = [x + 1 for x in params[param_key]]
    else:
        cron[rest_key] = params[param_key]