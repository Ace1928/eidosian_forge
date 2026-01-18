from __future__ import absolute_import, division, print_function
from datetime import datetime, timedelta
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible_collections.community.general.plugins.module_utils.rundeck import (
def job_status_check(self, execution_id):
    response = dict()
    timeout = False
    due = datetime.now() + timedelta(seconds=self.wait_execution_timeout)
    while not timeout:
        endpoint = 'execution/%d' % execution_id
        response = api_request(module=self.module, endpoint=endpoint)[0]
        output = api_request(module=self.module, endpoint='execution/%d/output' % execution_id)
        log_output = '\n'.join([x['log'] for x in output[0]['entries']])
        response.update({'output': log_output})
        if response['status'] == 'aborted':
            break
        elif response['status'] == 'scheduled':
            self.module.exit_json(msg='Job scheduled to run at %s' % self.run_at_time, execution_info=response, changed=True)
        elif response['status'] == 'failed':
            self.module.fail_json(msg='Job execution failed', execution_info=response)
        elif response['status'] == 'succeeded':
            self.module.exit_json(msg='Job execution succeeded!', execution_info=response)
        if datetime.now() >= due:
            timeout = True
            break
        sleep(self.wait_execution_delay)
    response.update({'timed_out': timeout})
    return response