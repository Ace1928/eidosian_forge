from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.csr_info import (
from ansible_collections.community.crypto.plugins.plugin_utils.filter_module import FilterModuleMock
def openssl_csr_info_filter(data, name_encoding='ignore'):
    """Extract information from X.509 PEM certificate."""
    if not isinstance(data, string_types):
        raise AnsibleFilterError('The community.crypto.openssl_csr_info input must be a text type, not %s' % type(data))
    if not isinstance(name_encoding, string_types):
        raise AnsibleFilterError('The name_encoding option must be of a text type, not %s' % type(name_encoding))
    name_encoding = to_native(name_encoding)
    if name_encoding not in ('ignore', 'idna', 'unicode'):
        raise AnsibleFilterError('The name_encoding option must be one of the values "ignore", "idna", or "unicode", not "%s"' % name_encoding)
    module = FilterModuleMock({'name_encoding': name_encoding})
    try:
        return get_csr_info(module, 'cryptography', content=to_bytes(data), validate_signature=True)
    except OpenSSLObjectError as exc:
        raise AnsibleFilterError(to_native(exc))