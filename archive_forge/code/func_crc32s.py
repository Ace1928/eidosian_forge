from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleFilterError
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.common.collections import is_string
def crc32s(value):
    if not is_string(value):
        raise AnsibleFilterError('Invalid value type (%s) for crc32 (%r)' % (type(value), value))
    if not HAS_ZLIB:
        raise AnsibleFilterError('Failed to import zlib module')
    data = to_bytes(value, errors='surrogate_or_strict')
    return '{0:x}'.format(crc32(data) & 4294967295)