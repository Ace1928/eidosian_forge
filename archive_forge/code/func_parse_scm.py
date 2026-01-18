from __future__ import (absolute_import, division, print_function)
import json
import os
import tarfile
import subprocess
import typing as t
from contextlib import contextmanager
from hashlib import sha256
from urllib.error import URLError
from urllib.parse import urldefrag
from shutil import rmtree
from tempfile import mkdtemp
from ansible.errors import AnsibleError
from ansible.galaxy import get_collections_galaxy_meta_info
from ansible.galaxy.api import should_retry_error
from ansible.galaxy.dependency_resolution.dataclasses import _GALAXY_YAML
from ansible.galaxy.user_agent import user_agent
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.api import retry_with_delays_and_condition
from ansible.module_utils.api import generate_jittered_backoff
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.yaml import yaml_load
from ansible.module_utils.urls import open_url
from ansible.utils.display import Display
from ansible.utils.sentinel import Sentinel
import yaml
def parse_scm(collection, version):
    """Extract name, version, path and subdir out of the SCM pointer."""
    if ',' in collection:
        collection, version = collection.split(',', 1)
    elif version == '*' or not version:
        version = 'HEAD'
    if collection.startswith('git+'):
        path = collection[4:]
    else:
        path = collection
    path, fragment = urldefrag(path)
    fragment = fragment.strip(os.path.sep)
    if path.endswith(os.path.sep + '.git'):
        name = path.split(os.path.sep)[-2]
    elif '://' not in path and '@' not in path:
        name = path
    else:
        name = path.split('/')[-1]
        if name.endswith('.git'):
            name = name[:-4]
    return (name, version, path, fragment)