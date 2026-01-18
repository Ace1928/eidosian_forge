from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import os
def local_file_path(module):
    if module.params['action'] == 'download':
        return module.params['dest']
    else:
        return module.params['src']