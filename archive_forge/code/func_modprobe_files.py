from __future__ import absolute_import, division, print_function
import os.path
import platform
import shlex
import traceback
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
@property
def modprobe_files(self):
    if not os.path.isdir(PARAMETERS_FILES_LOCATION):
        return []
    modules_paths = [os.path.join(PARAMETERS_FILES_LOCATION, path) for path in os.listdir(PARAMETERS_FILES_LOCATION)]
    return [path for path in modules_paths if os.path.isfile(path)]