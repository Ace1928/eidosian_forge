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
def install_src(collection, b_collection_path, b_collection_output_path, artifacts_manager):
    """Install the collection from source control into given dir.

    Generates the Ansible collection artifact data from a galaxy.yml and
    installs the artifact to a directory.
    This should follow the same pattern as build_collection, but instead
    of creating an artifact, install it.

    :param collection: Collection to be installed.
    :param b_collection_path: Collection dirs layout path.
    :param b_collection_output_path: The installation directory for the \\
                                     collection artifact.
    :param artifacts_manager: Artifacts manager.

    :raises AnsibleError: If no collection metadata found.
    """
    collection_meta = artifacts_manager.get_direct_collection_meta(collection)
    if 'build_ignore' not in collection_meta:
        collection_meta['build_ignore'] = []
        collection_meta['manifest'] = Sentinel
    collection_manifest = _build_manifest(**collection_meta)
    file_manifest = _build_files_manifest(b_collection_path, collection_meta['namespace'], collection_meta['name'], collection_meta['build_ignore'], collection_meta['manifest'], collection_meta['license_file'])
    collection_output_path = _build_collection_dir(b_collection_path, b_collection_output_path, collection_manifest, file_manifest)
    display.display('Created collection for {coll!s} at {path!s}'.format(coll=collection, path=collection_output_path))