from __future__ import absolute_import, division, print_function
from collections import defaultdict
import re
from ansible.module_utils.basic import AnsibleModule
def query_update(module, run_pkgng, name):
    rc, out, err = run_pkgng('upgrade', '-g', '-n', name)
    return rc == 1