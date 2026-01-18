from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def cast_connparams(connparams_dict):
    """Cast the passed connparams_dict dictionary

    Returns:
        Dictionary
    """
    for param, val in iteritems(connparams_dict):
        try:
            connparams_dict[param] = int(val)
        except ValueError:
            connparams_dict[param] = val
    return connparams_dict