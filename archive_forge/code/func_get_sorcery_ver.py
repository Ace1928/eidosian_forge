from __future__ import absolute_import, division, print_function
import datetime
import fileinput
import os
import re
import shutil
import sys
from ansible.module_utils.basic import AnsibleModule
def get_sorcery_ver(module):
    """ Get Sorcery version. """
    cmd_sorcery = '%s --version' % SORCERY['sorcery']
    rc, stdout, stderr = module.run_command(cmd_sorcery)
    if rc != 0 or not stdout:
        module.fail_json(msg='unable to get Sorcery version')
    return stdout.strip()