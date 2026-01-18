from __future__ import absolute_import, division, print_function
import errno
import fnmatch
import grp
import os
import pwd
import re
import stat
import time
from ansible.module_utils.common.text.converters import to_text, to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
def mode_filter(st, mode, exact, module):
    if not mode:
        return True
    st_mode = stat.S_IMODE(st.st_mode)
    try:
        mode = int(mode, 8)
    except ValueError:
        mode = module._symbolic_mode_to_octal(_Object(st_mode=0), mode)
    mode = stat.S_IMODE(mode)
    if exact:
        return st_mode == mode
    return bool(st_mode & mode)