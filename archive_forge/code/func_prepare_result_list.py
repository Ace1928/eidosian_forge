from __future__ import absolute_import, division, print_function
from ansible.module_utils.six.moves.urllib.parse import quote
from . import errors
def prepare_result_list(result):
    if isinstance(result, list):
        return result
    return [] if result is None else [result]