from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def find_hook(self, project, hook_url):
    for hook in project.hooks.list(**list_all_kwargs):
        if hook.url == hook_url:
            return hook