from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import re
def has_unknown_variable(self, out, err):
    return err.find('unknown variable') > 0 or out.find('unknown variable') > 0