from __future__ import (absolute_import, division, print_function)
import errno
import fnmatch
import functools
import json
import os
import pathlib
import queue
import re
import shutil
import stat
import sys
import tarfile
import tempfile
import textwrap
import threading
import time
import typing as t
from collections import namedtuple
from contextlib import contextmanager
from dataclasses import dataclass, fields as dc_fields
from hashlib import sha256
from io import BytesIO
from importlib.metadata import distribution
from itertools import chain
import ansible.constants as C
from ansible.compat.importlib_resources import files
from ansible.errors import AnsibleError
from ansible.galaxy.api import GalaxyAPI
from ansible.galaxy.collection.concrete_artifact_manager import (
from ansible.galaxy.collection.galaxy_api_proxy import MultiGalaxyAPIProxy
from ansible.galaxy.collection.gpg import (
from ansible.galaxy.dependency_resolution.dataclasses import (
from ansible.galaxy.dependency_resolution.versioning import meets_requirements
from ansible.plugins.loader import get_all_plugin_loaders
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.common.yaml import yaml_dump
from ansible.utils.collection_loader import AnsibleCollectionRef
from ansible.utils.display import Display
from ansible.utils.hashing import secure_hash, secure_hash_s
from ansible.utils.sentinel import Sentinel
def install_artifact(b_coll_targz_path, b_collection_path, b_temp_path, signatures, keyring, required_signature_count, ignore_signature_errors):
    """Install a collection from tarball under a given path.

    :param b_coll_targz_path: Collection tarball to be installed.
    :param b_collection_path: Collection dirs layout path.
    :param b_temp_path: Temporary dir path.
    :param signatures: frozenset of signatures to verify the MANIFEST.json
    :param keyring: The keyring used during GPG verification
    :param required_signature_count: The number of signatures that must successfully verify the collection
    :param ignore_signature_errors: GPG errors to ignore during signature verification
    """
    try:
        with tarfile.open(b_coll_targz_path, mode='r') as collection_tar:
            collection_tar._ansible_normalized_cache = {m.name.removesuffix(os.path.sep): m for m in collection_tar.getmembers()}
            _extract_tar_file(collection_tar, MANIFEST_FILENAME, b_collection_path, b_temp_path)
            if keyring is not None:
                manifest_file = os.path.join(to_text(b_collection_path, errors='surrogate_or_strict'), MANIFEST_FILENAME)
                verify_artifact_manifest(manifest_file, signatures, keyring, required_signature_count, ignore_signature_errors)
            files_member_obj = collection_tar.getmember('FILES.json')
            with _tarfile_extract(collection_tar, files_member_obj) as (dummy, files_obj):
                files = json.loads(to_text(files_obj.read(), errors='surrogate_or_strict'))
            _extract_tar_file(collection_tar, 'FILES.json', b_collection_path, b_temp_path)
            for file_info in files['files']:
                file_name = file_info['name']
                if file_name == '.':
                    continue
                if file_info['ftype'] == 'file':
                    _extract_tar_file(collection_tar, file_name, b_collection_path, b_temp_path, expected_hash=file_info['chksum_sha256'])
                else:
                    _extract_tar_dir(collection_tar, file_name, b_collection_path)
    except Exception:
        shutil.rmtree(b_collection_path)
        b_namespace_path = os.path.dirname(b_collection_path)
        if not os.listdir(b_namespace_path):
            os.rmdir(b_namespace_path)
        raise