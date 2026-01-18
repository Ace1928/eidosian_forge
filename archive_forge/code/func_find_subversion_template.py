from __future__ import (absolute_import, division, print_function)
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def find_subversion_template(module, templates_service):
    version = module.params.get('version')
    templates = templates_service.list()
    for template in templates:
        if version.get('number') == template.version.version_number and module.params.get('name') == template.name:
            return template
    raise ValueError("Template with name '%s' and version '%s' in cluster '%s' was not found'" % (module.params['name'], module.params['version']['number'], module.params['cluster']))