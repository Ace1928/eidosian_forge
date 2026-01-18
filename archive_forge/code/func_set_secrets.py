from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils.common.json import AnsibleJSONEncoder  # pylint: disable=unused-import
from ansible.parsing.vault import VaultLib
from ansible.parsing.yaml.objects import AnsibleVaultEncryptedUnicode
from ansible.utils.unsafe_proxy import wrap_var
@classmethod
def set_secrets(cls, secrets):
    cls._vaults['default'] = VaultLib(secrets=secrets)