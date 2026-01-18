from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def query_toplevel(module, name, world):
    regex = re.compile('^' + re.escape(name) + '([@=<>~].+)?$')
    with open(world) as f:
        content = f.read().split()
        for p in content:
            if regex.search(p):
                return True
    return False