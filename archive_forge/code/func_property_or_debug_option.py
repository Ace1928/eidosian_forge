import os
import sys
import debugpy
from debugpy import launcher
from debugpy.common import json
from debugpy.launcher import debuggee
def property_or_debug_option(prop_name, flag_name):
    assert prop_name[0].islower() and flag_name[0].isupper()
    value = request(prop_name, bool, optional=True)
    if value == ():
        value = None
    if flag_name in debug_options:
        if value is False:
            raise request.isnt_valid('{0}:false and "debugOptions":[{1}] are mutually exclusive', json.repr(prop_name), json.repr(flag_name))
        value = True
    return value