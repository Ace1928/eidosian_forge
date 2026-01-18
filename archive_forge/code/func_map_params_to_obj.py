from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_text
from ansible.module_utils.connection import Connection, ConnectionError
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves.urllib.parse import urlencode
def map_params_to_obj(module_params, key_transform):
    """The fn to convert the api returned params to module params
    :param module_params: Module params
    :param key_transform: Dict with module equivalent API params
    :rtype: A dict
    :returns: dict with module prams transformed having API expected params
    """
    obj = {}
    for k, v in iteritems(key_transform):
        if k in module_params and (module_params.get(k) or module_params.get(k) == 0 or module_params.get(k) is False):
            obj[v] = module_params.pop(k)
    return obj