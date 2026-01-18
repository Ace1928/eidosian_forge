from __future__ import (absolute_import, division, print_function)
import os
import time
from ansible.module_utils.basic import AnsibleModule
def expand_scheduled_tasks(eg, module):
    scheduled_tasks = module.params.get('scheduled_tasks')
    if scheduled_tasks is not None:
        eg_scheduling = spotinst.aws_elastigroup.Scheduling()
        eg_tasks = expand_list(scheduled_tasks, scheduled_task_fields, 'ScheduledTask')
        if len(eg_tasks) > 0:
            eg_scheduling.tasks = eg_tasks
            eg.scheduling = eg_scheduling