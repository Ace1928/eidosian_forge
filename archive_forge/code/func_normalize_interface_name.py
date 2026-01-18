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
def normalize_interface_name(configs, module, namekey=None):
    if not namekey:
        namekey = 'name'
    if configs:
        for conf in configs:
            if conf.get(namekey, None):
                conf[namekey] = get_normalize_interface_name(conf[namekey], module)