from __future__ import (absolute_import, division, print_function)
from yaml.constructor import SafeConstructor, ConstructorError
from yaml.nodes import MappingNode
from ansible import constants as C
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.parsing.yaml.objects import AnsibleMapping, AnsibleSequence, AnsibleUnicode, AnsibleVaultEncryptedUnicode
from ansible.parsing.vault import VaultLib
from ansible.utils.display import Display
from ansible.utils.unsafe_proxy import wrap_var
def construct_vault_encrypted_unicode(self, node):
    value = self.construct_scalar(node)
    b_ciphertext_data = to_bytes(value)
    vault = self._vaults['default']
    if vault.secrets is None:
        raise ConstructorError(context=None, context_mark=None, problem='found !vault but no vault password provided', problem_mark=node.start_mark, note=None)
    ret = AnsibleVaultEncryptedUnicode(b_ciphertext_data)
    ret.vault = vault
    ret.ansible_pos = self._node_position_info(node)
    return ret