from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def find_deploy_key(self, project, key_title):
    for deploy_key in project.keys.list(**list_all_kwargs):
        if deploy_key.title == key_title:
            return deploy_key