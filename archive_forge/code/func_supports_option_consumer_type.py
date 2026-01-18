from __future__ import absolute_import, division, print_function
from os.path import isfile
from os import getuid, unlink
import re
import shutil
import tempfile
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves import configparser
from ansible.module_utils import distro
def supports_option_consumer_type():
    if distro_id == 'fedora':
        return True
    if distro_id == 'rhel' and (distro_version[0] == 9 and distro_version[1] >= 2 or distro_version[0] >= 10):
        return True
    if distro_id == 'centos' and distro_version[0] >= 9:
        return True
    return False