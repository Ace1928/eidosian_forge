from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.basic import AnsibleModule
def get_local_version(pear_output):
    """Take pear remoteinfo output and get the installed version"""
    lines = pear_output.split('\n')
    for line in lines:
        if 'Installed ' in line:
            installed = line.rsplit(None, 1)[-1].strip()
            if installed == '-':
                continue
            return installed
    return None