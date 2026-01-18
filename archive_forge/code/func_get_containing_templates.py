from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_containing_templates(self, containing_templates):
    """
        Store tags from the playbook for template processing in DNAC.
        Check using check_return_status()

        Parameters:
            containing_templates (dict) - Containing tempaltes details
            containing Template information.

        Returns:
            containingTemplates (dict) - Organized containing templates parameters.
        """
    if containing_templates is None:
        return None
    containingTemplates = []
    i = 0
    for item in containing_templates:
        containingTemplates.append({})
        _tags = item.get('tags')
        if _tags is not None:
            containingTemplates[i].update({'tags': self.get_tags(_tags)})
        composite = item.get('composite')
        if composite is not None:
            containingTemplates[i].update({'composite': composite})
        description = item.get('description')
        if description is not None:
            containingTemplates[i].update({'description': description})
        device_types = item.get('device_types')
        if device_types is not None:
            containingTemplates[i].update({'deviceTypes': self.get_device_types(device_types)})
        id = item.get('id')
        if id is not None:
            containingTemplates[i].update({'id': id})
        name = item.get('name')
        if name is None:
            self.msg = 'name is mandatory under containing templates'
            self.status = 'failed'
            return self.check_return_status()
        containingTemplates[i].update({'name': name})
        language = item.get('language')
        if language is None:
            self.msg = 'language is mandatory under containing templates'
            self.status = 'failed'
            return self.check_return_status()
        language_list = ['JINJA', 'VELOCITY']
        if language not in language_list:
            self.msg = 'language under containing templates should be in ' + str(language_list)
            self.status = 'failed'
            return self.check_return_status()
        containingTemplates[i].update({'language': language})
        project_name = item.get('project_name')
        if project_name is not None:
            containingTemplates[i].update({'projectName': project_name})
        else:
            self.msg = 'project_name is mandatory under containing templates'
            self.status = 'failed'
            return self.check_return_status()
        rollback_template_params = item.get('rollback_template_params')
        if rollback_template_params is not None:
            containingTemplates[i].update({'rollbackTemplateParams': self.get_template_info(rollback_template_params)})
        template_content = item.get('template_content')
        if template_content is not None:
            containingTemplates[i].update({'templateContent': template_content})
        template_params = item.get('template_params')
        if template_params is not None:
            containingTemplates[i].update({'templateParams': self.get_template_info(template_params)})
        version = item.get('version')
        if version is not None:
            containingTemplates[i].update({'version': version})
    return containingTemplates