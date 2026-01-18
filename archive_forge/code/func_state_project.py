from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def state_project(self, state='active'):
    project = self.present_project()
    if project['state'].lower() != state:
        self.result['changed'] = True
        args = {'id': project['id']}
        if not self.module.check_mode:
            if state == 'suspended':
                project = self.query_api('suspendProject', **args)
            else:
                project = self.query_api('activateProject', **args)
            poll_async = self.module.params.get('poll_async')
            if project and poll_async:
                project = self.poll_job(project, 'project')
    return project