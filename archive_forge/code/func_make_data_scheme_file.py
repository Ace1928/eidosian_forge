import collections
import compileall
import contextlib
import csv
import importlib
import logging
import os.path
import re
import shutil
import sys
import warnings
from base64 import urlsafe_b64encode
from email.message import Message
from itertools import chain, filterfalse, starmap
from typing import (
from zipfile import ZipFile, ZipInfo
from pip._vendor.distlib.scripts import ScriptMaker
from pip._vendor.distlib.util import get_export_entry
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.exceptions import InstallationError
from pip._internal.locations import get_major_minor_version
from pip._internal.metadata import (
from pip._internal.models.direct_url import DIRECT_URL_METADATA_NAME, DirectUrl
from pip._internal.models.scheme import SCHEME_KEYS, Scheme
from pip._internal.utils.filesystem import adjacent_tmp_file, replace
from pip._internal.utils.misc import captured_stdout, ensure_dir, hash_file, partition
from pip._internal.utils.unpacking import (
from pip._internal.utils.wheel import parse_wheel
def make_data_scheme_file(record_path: RecordPath) -> 'File':
    normed_path = os.path.normpath(record_path)
    try:
        _, scheme_key, dest_subpath = normed_path.split(os.path.sep, 2)
    except ValueError:
        message = "Unexpected file in {}: {!r}. .data directory contents should be named like: '<scheme key>/<path>'.".format(wheel_path, record_path)
        raise InstallationError(message)
    try:
        scheme_path = scheme_paths[scheme_key]
    except KeyError:
        valid_scheme_keys = ', '.join(sorted(scheme_paths))
        message = 'Unknown scheme key used in {}: {} (for file {!r}). .data directory contents should be in subdirectories named with a valid scheme key ({})'.format(wheel_path, scheme_key, record_path, valid_scheme_keys)
        raise InstallationError(message)
    dest_path = os.path.join(scheme_path, dest_subpath)
    assert_no_path_traversal(scheme_path, dest_path)
    return ZipBackedFile(record_path, dest_path, zip_file)