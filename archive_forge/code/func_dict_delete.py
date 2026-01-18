from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import iteritems
def dict_delete(base, comparable):
    """
    This function generates a dict containing key, value pairs for keys
    that are present in the `base` dict but not present in the `comparable`
    dict.

    :param base: dict object to base the diff on
    :param comparable: dict object to compare against base
    :returns: new dict object with key, value pairs that needs to be deleted.

    """
    to_delete = dict()
    for key in base:
        if isinstance(base[key], dict):
            sub_diff = dict_delete(base[key], comparable.get(key, {}))
            if sub_diff:
                to_delete[key] = sub_diff
        elif key not in comparable:
            to_delete[key] = base[key]
    return to_delete