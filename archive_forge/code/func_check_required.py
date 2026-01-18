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
def check_required(module, required_parameters, parameters, options_context=None):
    """This utility is a wrapper for the Ansible "check_required_arguments"
    function. The "required_parameters" input list provides a list of
    key names that are required in the dictionary specified by "parameters".
    The optional "options_context" parameter specifies the context/path
    from the top level parent dict to the dict being checked."""
    if required_parameters:
        spec = {}
        for parameter in required_parameters:
            spec[parameter] = {'required': True}
        try:
            check_required_arguments(spec, parameters, options_context)
        except TypeError as exc:
            module.fail_json(msg=str(exc))