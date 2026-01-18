from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleFilterError
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import string_types
from ansible_collections.community.crypto.plugins.module_utils.serial import parse_serial
def parse_serial_filter(input):
    if not isinstance(input, string_types):
        raise AnsibleFilterError('The input for the community.crypto.parse_serial filter must be a string; got {type} instead'.format(type=type(input)))
    try:
        return parse_serial(to_native(input))
    except ValueError as exc:
        raise AnsibleFilterError(to_native(exc))