from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def exists_deploy_key(self, project, key_title):
    deploy_key = self.find_deploy_key(project, key_title)
    if deploy_key:
        self.deploy_key_object = deploy_key
        return True
    return False