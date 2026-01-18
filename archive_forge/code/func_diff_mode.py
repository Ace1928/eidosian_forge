from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.mh.exceptions import ModuleHelperException as _MHE
from ansible_collections.community.general.plugins.module_utils.mh.deco import module_fails_on_exception
@property
def diff_mode(self):
    return self.module._diff