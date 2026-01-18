from __future__ import (absolute_import, division, print_function)
import os
import stat
import re
def get_file_arg_spec():
    arg_spec = dict(mode=dict(type='raw'), owner=dict(), group=dict(), seuser=dict(), serole=dict(), selevel=dict(), setype=dict(), attributes=dict(aliases=['attr']))
    return arg_spec