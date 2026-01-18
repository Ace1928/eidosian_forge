from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_text
from ansible_collections.community.crypto.plugins.module_utils.crypto.pem import split_pem_list
def split_pem_filter(data):
    """Split PEM file."""
    if not isinstance(data, string_types):
        raise AnsibleFilterError('The community.crypto.split_pem input must be a text type, not %s' % type(data))
    data = to_text(data)
    return split_pem_list(data)