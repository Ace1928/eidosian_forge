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
def get_direct_collection_meta(self, collection):
    """Extract meta from the given on-disk collection artifact."""
    try:
        return self._artifact_meta_cache[collection.src]
    except KeyError:
        b_artifact_path = self.get_artifact_path(collection)
    if collection.is_url or collection.is_file:
        collection_meta = _get_meta_from_tar(b_artifact_path)
    elif collection.is_dir:
        try:
            collection_meta = _get_meta_from_dir(b_artifact_path, self.require_build_metadata)
        except LookupError as lookup_err:
            raise AnsibleError('Failed to find the collection dir deps: {err!s}'.format(err=to_native(lookup_err))) from lookup_err
    elif collection.is_scm:
        collection_meta = {'name': None, 'namespace': None, 'dependencies': {to_native(b_artifact_path): '*'}, 'version': '*'}
    elif collection.is_subdirs:
        collection_meta = {'name': None, 'namespace': None, 'dependencies': dict.fromkeys(map(to_native, collection.namespace_collection_paths), '*'), 'version': '*'}
    else:
        raise RuntimeError
    self._artifact_meta_cache[collection.src] = collection_meta
    return collection_meta