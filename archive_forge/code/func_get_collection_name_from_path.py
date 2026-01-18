import json
import os
import os.path
import sys
import tempfile
import ansible.module_utils.basic
from .exceptions import (
import ansible_collections.cloud.common.plugins.module_utils.turbo.common
def get_collection_name_from_path():
    module_path = ansible.module_utils.basic.get_module_path()
    ansiblez = module_path.split('/')[-3]
    if ansiblez.startswith('ansible_') and ansiblez.endswith('.zip'):
        return '.'.join(ansiblez[8:].split('.')[:2])