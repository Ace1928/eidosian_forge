from __future__ import (absolute_import, division, print_function)
import json
import os
import re
import yaml
from ansible.errors import AnsibleLookupError
from ansible.module_utils.compat.importlib import import_module
from ansible.plugins.lookup import LookupBase
def load_collection_meta(collection_pkg, no_version='*'):
    path = os.path.dirname(collection_pkg.__file__)
    manifest_path = os.path.join(path, 'MANIFEST.json')
    if os.path.exists(manifest_path):
        return load_collection_meta_manifest(manifest_path)
    galaxy_path = os.path.join(path, 'galaxy.yml')
    if os.path.exists(galaxy_path):
        return load_collection_meta_galaxy(galaxy_path, no_version=no_version)
    return {}