from __future__ import absolute_import, division, print_function
from ansible.errors import (
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.collections import is_sequence
def initialize_hashids(**kwargs):
    if not HAS_HASHIDS:
        raise AnsibleError('The hashids library must be installed in order to use this plugin')
    params = dict(((k, v) for k, v in kwargs.items() if v))
    try:
        return Hashids(**params)
    except TypeError as e:
        raise AnsibleFilterError('The provided parameters %s are invalid: %s' % (', '.join(['%s=%s' % (k, v) for k, v in params.items()]), to_native(e)))