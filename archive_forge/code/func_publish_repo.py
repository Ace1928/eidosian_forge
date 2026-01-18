from __future__ import absolute_import, division, print_function
import json
import os
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.urls import url_argument_spec
def publish_repo(self, repo_id, publish_distributor):
    url = '%s/pulp/api/v2/repositories/%s/actions/publish/' % (self.host, repo_id)
    if publish_distributor is None:
        repo_config = self.get_repo_config_by_id(repo_id)
        for distributor in repo_config['distributors']:
            data = dict()
            data['id'] = distributor['id']
            response, info = fetch_url(self.module, url, data=json.dumps(data), method='POST')
            if info['status'] != 202:
                self.module.fail_json(msg='Failed to publish the repo.', status_code=info['status'], response=info['msg'], url=url, distributor=distributor['id'])
    else:
        data = dict()
        data['id'] = publish_distributor
        response, info = fetch_url(self.module, url, data=json.dumps(data), method='POST')
        if info['status'] != 202:
            self.module.fail_json(msg='Failed to publish the repo', status_code=info['status'], response=info['msg'], url=url, distributor=publish_distributor)
    if self.wait_for_completion:
        self.verify_tasks_completed(json.load(response))
    return True