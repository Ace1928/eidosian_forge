from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def delete_project_or_template(self, config, is_delete_project=False):
    """
        Call DNAC API to delete project or template with provided inputs.

        Parameters:
            config (dict) - Playbook details containing template information.
            is_delete_project (bool) - True if we need to delete project, else False.

        Returns:
            self
        """
    if is_delete_project:
        params_key = {'project_id': self.have_project.get('id')}
        deletion_value = 'deletes_the_project'
        name = 'project: {0}'.format(config.get('configuration_templates').get('project_name'))
    else:
        template_params = self.want.get('template_params')
        params_key = {'template_id': self.have_template.get('id')}
        deletion_value = 'deletes_the_template'
        name = 'templateName: {0}'.format(template_params.get('templateName'))
    response = self.dnac_apply['exec'](family='configuration_templates', function=deletion_value, params=params_key)
    task_id = response.get('response').get('taskId')
    if task_id:
        task_details = self.get_task_details(task_id)
        self.result['changed'] = True
        self.result['msg'] = task_details.get('progress')
        self.result['diff'] = config.get('configuration_templates')
        self.log("Task details for '{0}': {1}".format(deletion_value, task_details), 'DEBUG')
        self.result['response'] = task_details if task_details else response
        if not self.result['msg']:
            self.result['msg'] = 'Error while deleting {name} : '
            self.status = 'failed'
            return self
    self.msg = 'Successfully deleted {0} '.format(name)
    self.status = 'success'
    return self