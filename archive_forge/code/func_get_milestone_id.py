from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.api import basic_auth_argument_spec
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible_collections.community.general.plugins.module_utils.gitlab import (
from datetime import datetime
def get_milestone_id(self, _title):
    _milestone_list = self.gitlab_object.milestones.list()
    _found = list(filter(lambda x: x.title == _title, _milestone_list))
    if _found:
        return _found[0].id
    else:
        self._module.fail_json(msg="milestone '%s' not found." % _title)