from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
def get_vault_diff(client, ipa_vault, module_vault, module):
    return client.get_diff(ipa_data=ipa_vault, module_data=module_vault)