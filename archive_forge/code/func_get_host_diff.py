from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
def get_host_diff(client, ipa_host, module_host):
    non_updateable_keys = ['force', 'ip_address']
    if not module_host.get('random'):
        non_updateable_keys.append('random')
    for key in non_updateable_keys:
        if key in module_host:
            del module_host[key]
    return client.get_diff(ipa_data=ipa_host, module_data=module_host)