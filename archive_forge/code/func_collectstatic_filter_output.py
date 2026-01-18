from __future__ import absolute_import, division, print_function
import os
import sys
import shlex
from ansible.module_utils.basic import AnsibleModule
def collectstatic_filter_output(line):
    return line and '0 static files' not in line