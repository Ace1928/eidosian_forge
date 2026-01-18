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
def is_entrypoint_wrapper(file: 'File') -> bool:
    path = file.dest_path
    name = os.path.basename(path)
    if name.lower().endswith('.exe'):
        matchname = name[:-4]
    elif name.lower().endswith('-script.py'):
        matchname = name[:-10]
    elif name.lower().endswith('.pya'):
        matchname = name[:-4]
    else:
        matchname = name
    return matchname in console or matchname in gui