from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def issue_has_changed(self, issue, options):
    for key, value in options.items():
        if value is not None:
            if key == 'milestone_id':
                old_milestone = getattr(issue, 'milestone')['id'] if getattr(issue, 'milestone') else ''
                if options[key] != old_milestone:
                    return True
            elif key == 'assignee_ids':
                if options[key] != sorted([user['id'] for user in getattr(issue, 'assignees')]):
                    return True
            elif key == 'labels':
                if options[key] != sorted(getattr(issue, key)):
                    return True
            elif getattr(issue, key) != value:
                return True
    return False