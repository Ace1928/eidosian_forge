from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def full_upgrade_packages(module):
    do_upgrade_packages(module, True)