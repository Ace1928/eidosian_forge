from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import env_fallback
from ansible.module_utils._text import to_native
import os.path
def delete_default_rule(ls):
    index = 0
    for elem in ls:
        if elem['comment'].lower() == 'default rule':
            del ls[index]
            break
        index = index + 1
    print(ls)
    return ls