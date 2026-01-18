from __future__ import (absolute_import, division, print_function)
import csv
import os
import json
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_all_data_with_pagination, strip_substr_dict
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.common.dict_transformations import recursive_diff
from datetime import datetime
def get_schedule_payload(module):
    schedule_payload = {}
    inp_schedule = module.params.get('date_and_time')
    if inp_schedule:
        time_interval = bool(inp_schedule.get('time_interval'))
        schedule_payload['Interval'] = time_interval
        schedule_payload['StartTime'], start_time_x = get_ftime(module, inp_schedule, 'from', time_interval)
        schedule_payload['EndTime'], end_time_x = get_ftime(module, inp_schedule, 'to', time_interval)
        if inp_schedule.get('date_to') and end_time_x < start_time_x:
            module.exit_json(failed=True, msg=END_START_TIME.format(end_time_x, start_time_x))
        weekdays = {'monday': 'mon', 'tuesday': 'tue', 'wednesday': 'wed', 'thursday': 'thu', 'friday': 'fri', 'saturday': 'sat', 'sunday': 'sun'}
        inp_week_list = ['*']
        cron_sep = ','
        if inp_schedule.get('days'):
            week_order = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            inp_week_list = sorted(list(set(inp_schedule.get('days'))), key=week_order.index)
        schedule_payload['CronString'] = f'* * * ? * {cron_sep.join([weekdays.get(x, '*') for x in inp_week_list])} *'
    return {'Schedule': schedule_payload} if schedule_payload else {}