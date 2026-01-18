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
def install_collections(collections, output_path, apis, ignore_errors, no_deps, force, force_deps, upgrade, allow_pre_release, artifacts_manager, disable_gpg_verify, offline, read_requirement_paths):
    """Install Ansible collections to the path specified.

    :param collections: The collections to install.
    :param output_path: The path to install the collections to.
    :param apis: A list of GalaxyAPIs to query when searching for a collection.
    :param validate_certs: Whether to validate the certificates if downloading a tarball.
    :param ignore_errors: Whether to ignore any errors when installing the collection.
    :param no_deps: Ignore any collection dependencies and only install the base requirements.
    :param force: Re-install a collection if it has already been installed.
    :param force_deps: Re-install a collection as well as its dependencies if they have already been installed.
    """
    existing_collections = {Requirement(coll.fqcn, coll.ver, coll.src, coll.type, None) for path in {output_path} | read_requirement_paths for coll in find_existing_collections(path, artifacts_manager)}
    unsatisfied_requirements = set(chain.from_iterable(((Requirement.from_dir_path(to_bytes(sub_coll), artifacts_manager) for sub_coll in artifacts_manager.get_direct_collection_dependencies(install_req).keys()) if install_req.is_subdirs else (install_req,) for install_req in collections)))
    requested_requirements_names = {req.fqcn for req in unsatisfied_requirements}
    unsatisfied_requirements -= set() if force or force_deps else {req for req in unsatisfied_requirements for exs in existing_collections if req.fqcn == exs.fqcn and meets_requirements(exs.ver, req.ver)}
    if not unsatisfied_requirements and (not upgrade):
        display.display('Nothing to do. All requested collections are already installed. If you want to reinstall them, consider using `--force`.')
        return
    existing_non_requested_collections = {coll for coll in existing_collections if coll.fqcn not in requested_requirements_names}
    preferred_requirements = [] if force_deps else existing_non_requested_collections if force else existing_collections
    preferred_collections = {Candidate(coll.fqcn, coll.ver, coll.src, coll.type, None) for coll in preferred_requirements}
    with _display_progress('Process install dependency map'):
        dependency_map = _resolve_depenency_map(collections, galaxy_apis=apis, preferred_candidates=preferred_collections, concrete_artifacts_manager=artifacts_manager, no_deps=no_deps, allow_pre_release=allow_pre_release, upgrade=upgrade, include_signatures=not disable_gpg_verify, offline=offline)
    keyring_exists = artifacts_manager.keyring is not None
    with _display_progress('Starting collection install process'):
        for fqcn, concrete_coll_pin in dependency_map.items():
            if concrete_coll_pin.is_virtual:
                display.vvvv("'{coll!s}' is virtual, skipping.".format(coll=to_text(concrete_coll_pin)))
                continue
            if concrete_coll_pin in preferred_collections:
                display.display("'{coll!s}' is already installed, skipping.".format(coll=to_text(concrete_coll_pin)))
                continue
            if not disable_gpg_verify and concrete_coll_pin.signatures and (not keyring_exists):
                display.warning('The GnuPG keyring used for collection signature verification was not configured but signatures were provided by the Galaxy server to verify authenticity. Configure a keyring for ansible-galaxy to use or disable signature verification. Skipping signature verification.')
            if concrete_coll_pin.type == 'galaxy':
                concrete_coll_pin = concrete_coll_pin.with_signatures_repopulated()
            try:
                install(concrete_coll_pin, output_path, artifacts_manager)
            except AnsibleError as err:
                if ignore_errors:
                    display.warning('Failed to install collection {coll!s} but skipping due to --ignore-errors being set. Error: {error!s}'.format(coll=to_text(concrete_coll_pin), error=to_text(err)))
                else:
                    raise