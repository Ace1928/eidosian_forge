from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def handle_import(self, config):
    """
        Import templates and projects in DNAC with fields provided in DNAC.

        Parameters:
            config (dict) - Playbook details containing template information.

        Returns:
            self
        """
    _import = config.get('import')
    if _import:
        do_version = _import.get('project').get('do_version')
        payload = None
        if _import.get('project').get('payload'):
            payload = _import.get('project').get('payload')
        else:
            self.msg = 'Mandatory parameter payload is not found under import project'
            self.status = 'failed'
            return self
        _import_project = {'doVersion': do_version, 'payload': payload}
        self.log('Importing project details from the playbook: {0}'.format(_import_project), 'DEBUG')
        if _import_project:
            response = self.dnac._exec(family='configuration_templates', function='imports_the_projects_provided', params=_import_project)
            validation_string = 'successfully imported project'
            self.check_task_response_status(response, validation_string).check_return_status()
            self.result['response'][0].update({'importProject': validation_string})
        _import_template = _import.get('template')
        if _import_template.get('project_name'):
            self.msg = 'Mandatory paramter project_name is not found under import template'
            self.status = 'failed'
            return self
        if _import_template.get('payload'):
            self.msg = 'Mandatory paramter payload is not found under import template'
            self.status = 'failed'
            return self
        payload = _import_template.get('project_name')
        import_template = {'doVersion': _import_template.get('do_version'), 'projectName': _import_template.get('project_name'), 'payload': self.get_template_params(payload)}
        self.log('Import template details from the playbook: {0}'.format(_import_template), 'DEBUG')
        if _import_template:
            response = self.dnac._exec(family='configuration_templates', function='imports_the_templates_provided', params=import_template)
            validation_string = 'successfully imported template'
            self.check_task_response_status(response, validation_string).check_return_status()
            self.result['response'][0].update({'importTemplate': validation_string})
    return self