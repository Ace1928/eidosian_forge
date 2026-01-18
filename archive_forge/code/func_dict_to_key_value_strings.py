from __future__ import absolute_import, division, print_function
from ansible.module_utils.six.moves.urllib.parse import quote
from . import errors
def dict_to_key_value_strings(data):
    return ['{0}={1}'.format(k, v) for k, v in data.items()]