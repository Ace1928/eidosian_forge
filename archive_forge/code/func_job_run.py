from __future__ import absolute_import, division, print_function
from datetime import datetime, timedelta
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible_collections.community.general.plugins.module_utils.rundeck import (
def job_run(self):
    response, info = api_request(module=self.module, endpoint='job/%s/run' % quote(self.job_id), method='POST', data={'loglevel': self.loglevel, 'options': self.job_options, 'runAtTime': self.run_at_time, 'filter': self.filter_nodes})
    if info['status'] != 200:
        self.module.fail_json(msg=info['msg'])
    if not self.wait_execution:
        self.module.exit_json(msg='Job run send successfully!', execution_info=response)
    job_status = self.job_status_check(response['id'])
    if job_status['timed_out']:
        if self.abort_on_timeout:
            api_request(module=self.module, endpoint='execution/%s/abort' % response['id'], method='GET')
            abort_status = self.job_status_check(response['id'])
            self.module.fail_json(msg='Job execution aborted due the timeout specified', execution_info=abort_status)
        self.module.fail_json(msg='Job execution timed out', execution_info=job_status)