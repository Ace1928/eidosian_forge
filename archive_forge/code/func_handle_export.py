from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def handle_export(self, config):
    """
        Export templates and projects in DNAC with fields provided in DNAC.

        Parameters:
            config (dict) - Playbook details containing template information.

        Returns:
            self
        """
    export = config.get('export')
    if export:
        export_project = export.get('project')
        self.log('Export project playbook details: {0}'.format(export_project), 'DEBUG')
        if export_project:
            response = self.dnac._exec(family='configuration_templates', function='export_projects', params={'payload': export_project})
            validation_string = 'successfully exported project'
            self.check_task_response_status(response, validation_string, True).check_return_status()
            self.result['response'][0].update({'exportProject': self.msg})
        export_values = export.get('template')
        if export_values:
            self.get_export_template_values(export_values).check_return_status()
            self.log('Exporting template playbook details: {0}'.format(self.export_template), 'DEBUG')
            response = self.dnac._exec(family='configuration_templates', function='export_templates', params={'payload': self.export_template})
            validation_string = 'successfully exported template'
            self.check_task_response_status(response, validation_string, True).check_return_status()
            self.result['response'][0].update({'exportTemplate': self.msg})
    return self