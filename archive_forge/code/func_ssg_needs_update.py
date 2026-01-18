from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.error import HTTPError
@property
def ssg_needs_update(self):
    if self.ssg_data['fullWarnThreshold'] != self.warning_threshold or self.ssg_data['autoDeleteLimit'] != self.delete_limit or self.ssg_data['repFullPolicy'] != self.full_policy or (self.ssg_data['rollbackPriority'] != self.rollback_priority):
        return True
    else:
        return False