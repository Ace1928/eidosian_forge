from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError, AnsibleFilterError
def json_query(data, expr):
    """Query data using jmespath query language ( http://jmespath.org ). Example:
    - ansible.builtin.debug: msg="{{ instance | json_query(tagged_instances[*].block_device_mapping.*.volume_id') }}"
    """
    if not HAS_LIB:
        raise AnsibleError('You need to install "jmespath" prior to running json_query filter')
    jmespath.functions.REVERSE_TYPES_MAP['string'] = jmespath.functions.REVERSE_TYPES_MAP['string'] + ('AnsibleUnicode', 'AnsibleUnsafeText')
    jmespath.functions.REVERSE_TYPES_MAP['array'] = jmespath.functions.REVERSE_TYPES_MAP['array'] + ('AnsibleSequence',)
    jmespath.functions.REVERSE_TYPES_MAP['object'] = jmespath.functions.REVERSE_TYPES_MAP['object'] + ('AnsibleMapping',)
    try:
        return jmespath.search(expr, data)
    except jmespath.exceptions.JMESPathError as e:
        raise AnsibleFilterError('JMESPathError in json_query filter plugin:\n%s' % e)
    except Exception as e:
        raise AnsibleFilterError('Error in jmespath.search in json_query filter plugin:\n%s' % e)