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
def build_collection(u_collection_path, u_output_path, force):
    """Creates the Ansible collection artifact in a .tar.gz file.

    :param u_collection_path: The path to the collection to build. This should be the directory that contains the
        galaxy.yml file.
    :param u_output_path: The path to create the collection build artifact. This should be a directory.
    :param force: Whether to overwrite an existing collection build artifact or fail.
    :return: The path to the collection build artifact.
    """
    b_collection_path = to_bytes(u_collection_path, errors='surrogate_or_strict')
    try:
        collection_meta = _get_meta_from_src_dir(b_collection_path)
    except LookupError as lookup_err:
        raise AnsibleError(to_native(lookup_err)) from lookup_err
    collection_manifest = _build_manifest(**collection_meta)
    file_manifest = _build_files_manifest(b_collection_path, collection_meta['namespace'], collection_meta['name'], collection_meta['build_ignore'], collection_meta['manifest'], collection_meta['license_file'])
    artifact_tarball_file_name = '{ns!s}-{name!s}-{ver!s}.tar.gz'.format(name=collection_meta['name'], ns=collection_meta['namespace'], ver=collection_meta['version'])
    b_collection_output = os.path.join(to_bytes(u_output_path), to_bytes(artifact_tarball_file_name, errors='surrogate_or_strict'))
    if os.path.exists(b_collection_output):
        if os.path.isdir(b_collection_output):
            raise AnsibleError("The output collection artifact '%s' already exists, but is a directory - aborting" % to_native(b_collection_output))
        elif not force:
            raise AnsibleError("The file '%s' already exists. You can use --force to re-create the collection artifact." % to_native(b_collection_output))
    collection_output = _build_collection_tar(b_collection_path, b_collection_output, collection_manifest, file_manifest)
    return collection_output