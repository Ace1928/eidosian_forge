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
def get_entrypoints(dist: BaseDistribution) -> Tuple[Dict[str, str], Dict[str, str]]:
    console_scripts = {}
    gui_scripts = {}
    for entry_point in dist.iter_entry_points():
        if entry_point.group == 'console_scripts':
            console_scripts[entry_point.name] = entry_point.value
        elif entry_point.group == 'gui_scripts':
            gui_scripts[entry_point.name] = entry_point.value
    return (console_scripts, gui_scripts)