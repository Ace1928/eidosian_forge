from __future__ import (absolute_import, division, print_function)
import os
import re
from uuid import UUID
from ansible.module_utils.six import text_type, binary_type
def rax_scaling_group_personality_file(module, files):
    if not files:
        return []
    results = []
    for rpath, lpath in files.items():
        lpath = os.path.expanduser(lpath)
        try:
            with open(lpath, 'r') as f:
                results.append({'path': rpath, 'contents': f.read()})
        except Exception as e:
            module.fail_json(msg='Failed to load %s: %s' % (lpath, str(e)))
    return results