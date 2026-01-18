from __future__ import absolute_import, division, print_function
import json
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import missing_required_lib
from ..module_utils.gcp_utils import (
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
def projects_for_folder(self, config_data, folder):
    link = 'https://cloudresourcemanager.googleapis.com/v1/projects'
    query = 'parent.id = {0}'.format(folder)
    projects = []
    config_data['scopes'] = ['https://www.googleapis.com/auth/cloud-platform']
    projects_response = self.fetch_projects(config_data, link, query)
    if 'projects' in projects_response:
        for item in projects_response.get('projects'):
            projects.append(item['projectId'])
    return projects