from __future__ import absolute_import, division, print_function
import re
import json
import ast
from copy import copy
from itertools import (count, groupby)
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible.module_utils.common.network import (
from ansible.module_utils.common.validation import check_required_arguments
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def remove_empties_from_list(config_list):
    ret_config = []
    if not config_list or not isinstance(config_list, list):
        return ret_config
    for config in config_list:
        if isinstance(config, dict):
            ret_config.append(remove_empties(config))
        else:
            ret_config.append(copy(config))
    return ret_config