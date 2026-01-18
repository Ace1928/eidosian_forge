from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def sudorule_add_deny_command_group(self, name, item):
    return self._post_json(method='sudorule_add_deny_command', name=name, item={'sudocmdgroup': item})