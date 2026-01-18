from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError
from ansible.utils.display import Display
from ansible_collections.community.hashi_vault.plugins.plugin_utils._hashi_vault_lookup_base import HashiVaultLookupBase
from ansible_collections.community.hashi_vault.plugins.module_utils._hashi_vault_common import HashiVaultValueError
def field_ops(self):
    secret = self.get_option('secret')
    s_f = secret.rsplit(':', 1)
    self.set_option('secret', s_f[0])
    if len(s_f) >= 2:
        field = s_f[1]
    else:
        field = None
    self._secret_field = field