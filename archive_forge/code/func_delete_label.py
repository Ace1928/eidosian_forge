from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.api import basic_auth_argument_spec
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def delete_label(self, var_obj):
    if self._module.check_mode:
        return (True, True)
    _label = self.gitlab_object.labels.get(var_obj.get('name'))
    _label.delete()
    return (True, _label.asdict())