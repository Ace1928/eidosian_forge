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
def get_galaxy_artifact_path(self, collection):
    """Given a Galaxy-stored collection, return a cached path.

        If it's not yet on disk, this method downloads the artifact first.
        """
    try:
        return self._galaxy_artifact_cache[collection]
    except KeyError:
        pass
    try:
        url, sha256_hash, token = self._galaxy_collection_cache[collection]
    except KeyError as key_err:
        raise RuntimeError('There is no known source for {coll!s}'.format(coll=collection)) from key_err
    display.vvvv("Fetching a collection tarball for '{collection!s}' from Ansible Galaxy".format(collection=collection))
    try:
        b_artifact_path = _download_file(url, self._b_working_directory, expected_hash=sha256_hash, validate_certs=self._validate_certs, token=token)
    except URLError as err:
        raise AnsibleError("Failed to download collection tar from '{coll_src!s}': {download_err!s}".format(coll_src=to_native(collection.src), download_err=to_native(err))) from err
    except Exception as err:
        raise AnsibleError("Failed to download collection tar from '{coll_src!s}' due to the following unforeseen error: {download_err!s}".format(coll_src=to_native(collection.src), download_err=to_native(err))) from err
    else:
        display.vvv("Collection '{coll!s}' obtained from server {server!s} {url!s}".format(coll=collection, server=collection.src or 'Galaxy', url=collection.src.api_server if collection.src is not None else ''))
    self._galaxy_artifact_cache[collection] = b_artifact_path
    return b_artifact_path