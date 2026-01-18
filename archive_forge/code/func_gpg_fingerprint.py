from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleFilterError
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.module_utils.six import string_types
from ansible_collections.community.crypto.plugins.module_utils.gnupg.cli import GPGError, get_fingerprint_from_bytes
from ansible_collections.community.crypto.plugins.plugin_utils.gnupg import PluginGPGRunner
def gpg_fingerprint(input):
    if not isinstance(input, string_types):
        raise AnsibleFilterError('The input for the community.crypto.gpg_fingerprint filter must be a string; got {type} instead'.format(type=type(input)))
    try:
        gpg = PluginGPGRunner()
        return get_fingerprint_from_bytes(gpg, to_bytes(input))
    except GPGError as exc:
        raise AnsibleFilterError(to_native(exc))