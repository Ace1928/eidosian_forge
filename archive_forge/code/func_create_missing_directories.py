from __future__ import absolute_import, division, print_function
import os
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils import deps
from ansible.module_utils.common.text.converters import to_native
def create_missing_directories(dest):
    destpath = os.path.dirname(dest)
    if not os.path.exists(destpath):
        os.makedirs(destpath)