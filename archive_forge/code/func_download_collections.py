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
def download_collections(collections, output_path, apis, no_deps, allow_pre_release, artifacts_manager):
    """Download Ansible collections as their tarball from a Galaxy server to the path specified and creates a requirements
    file of the downloaded requirements to be used for an install.

    :param collections: The collections to download, should be a list of tuples with (name, requirement, Galaxy Server).
    :param output_path: The path to download the collections to.
    :param apis: A list of GalaxyAPIs to query when search for a collection.
    :param validate_certs: Whether to validate the certificate if downloading a tarball from a non-Galaxy host.
    :param no_deps: Ignore any collection dependencies and only download the base requirements.
    :param allow_pre_release: Do not ignore pre-release versions when selecting the latest.
    """
    with _display_progress('Process download dependency map'):
        dep_map = _resolve_depenency_map(set(collections), galaxy_apis=apis, preferred_candidates=None, concrete_artifacts_manager=artifacts_manager, no_deps=no_deps, allow_pre_release=allow_pre_release, upgrade=False, include_signatures=False, offline=False)
    b_output_path = to_bytes(output_path, errors='surrogate_or_strict')
    requirements = []
    with _display_progress("Starting collection download process to '{path!s}'".format(path=output_path)):
        for fqcn, concrete_coll_pin in dep_map.copy().items():
            if concrete_coll_pin.is_virtual:
                display.display('Virtual collection {coll!s} is not downloadable'.format(coll=to_text(concrete_coll_pin)))
                continue
            display.display(u"Downloading collection '{coll!s}' to '{path!s}'".format(coll=to_text(concrete_coll_pin), path=to_text(b_output_path)))
            b_src_path = artifacts_manager.get_artifact_path_from_unknown(concrete_coll_pin)
            b_dest_path = os.path.join(b_output_path, os.path.basename(b_src_path))
            if concrete_coll_pin.is_dir:
                b_dest_path = to_bytes(build_collection(to_text(b_src_path, errors='surrogate_or_strict'), to_text(output_path, errors='surrogate_or_strict'), force=True), errors='surrogate_or_strict')
            else:
                shutil.copy(to_native(b_src_path), to_native(b_dest_path))
            display.display("Collection '{coll!s}' was downloaded successfully".format(coll=concrete_coll_pin))
            requirements.append({'name': to_native(os.path.basename(b_dest_path)), 'version': concrete_coll_pin.ver})
        requirements_path = os.path.join(output_path, 'requirements.yml')
        b_requirements_path = to_bytes(requirements_path, errors='surrogate_or_strict')
        display.display(u"Writing requirements.yml file of downloaded collections to '{path!s}'".format(path=to_text(requirements_path)))
        yaml_bytes = to_bytes(yaml_dump({'collections': requirements}), errors='surrogate_or_strict')
        with open(b_requirements_path, mode='wb') as req_fd:
            req_fd.write(yaml_bytes)