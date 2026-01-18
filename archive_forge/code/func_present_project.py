from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def present_project(self):
    project = self.get_project()
    if not project:
        project = self.create_project(project)
    else:
        project = self.update_project(project)
    if project:
        project = self.ensure_tags(resource=project, resource_type='project')
        self.project = project
    return project