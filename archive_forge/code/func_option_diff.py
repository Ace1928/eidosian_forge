from __future__ import absolute_import, division, print_function
import atexit
import ansible.module_utils.common._collections_compat as collections_compat
import json
import os
import re
import socket
import ssl
import hashlib
import time
import traceback
import datetime
from collections import OrderedDict
from ansible.module_utils.compat.version import StrictVersion
from random import randint
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.six import integer_types, iteritems, string_types, raise_from
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.six.moves.urllib.parse import unquote
def option_diff(options, current_options, truthy_strings_as_bool=True):
    current_options_dict = {}
    for option in current_options:
        current_options_dict[option.key] = option.value
    change_option_list = []
    for option_key, option_value in options.items():
        if truthy_strings_as_bool and is_boolean(option_value):
            option_value = VmomiSupport.vmodlTypes['bool'](is_truthy(option_value))
        elif type(option_value) is int:
            option_value = VmomiSupport.vmodlTypes['int'](option_value)
        elif type(option_value) is float:
            option_value = VmomiSupport.vmodlTypes['float'](option_value)
        elif type(option_value) is str:
            option_value = VmomiSupport.vmodlTypes['string'](option_value)
        if option_key not in current_options_dict or current_options_dict[option_key] != option_value:
            change_option_list.append(vim.option.OptionValue(key=option_key, value=option_value))
    return change_option_list