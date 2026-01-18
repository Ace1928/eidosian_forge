from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.publickey_info import (
from ansible_collections.community.crypto.plugins.plugin_utils.filter_module import FilterModuleMock
def openssl_publickey_info_filter(data):
    """Extract information from OpenSSL PEM public key."""
    if not isinstance(data, string_types):
        raise AnsibleFilterError('The community.crypto.openssl_publickey_info input must be a text type, not %s' % type(data))
    module = FilterModuleMock({})
    try:
        return get_publickey_info(module, 'cryptography', content=to_bytes(data))
    except PublicKeyParseError as exc:
        raise AnsibleFilterError(exc.error_message)
    except OpenSSLObjectError as exc:
        raise AnsibleFilterError(to_native(exc))