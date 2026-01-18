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
def get_galaxy_artifact_source_info(self, collection):
    server = collection.src.api_server
    try:
        download_url = self._galaxy_collection_cache[collection][0]
        signatures_url, signatures = self._galaxy_collection_origin_cache[collection]
    except KeyError as key_err:
        raise RuntimeError('The is no known source for {coll!s}'.format(coll=collection)) from key_err
    return {'format_version': '1.0.0', 'namespace': collection.namespace, 'name': collection.name, 'version': collection.ver, 'server': server, 'version_url': signatures_url, 'download_url': download_url, 'signatures': signatures}