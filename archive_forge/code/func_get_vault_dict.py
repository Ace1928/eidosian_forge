from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
def get_vault_dict(description=None, vault_type=None, vault_salt=None, vault_public_key=None, service=None):
    vault = {}
    if description is not None:
        vault['description'] = description
    if vault_type is not None:
        vault['ipavaulttype'] = vault_type
    if vault_salt is not None:
        vault['ipavaultsalt'] = vault_salt
    if vault_public_key is not None:
        vault['ipavaultpublickey'] = vault_public_key
    if service is not None:
        vault['service'] = service
    return vault