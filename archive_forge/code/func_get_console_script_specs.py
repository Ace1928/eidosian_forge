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
def get_console_script_specs(console: Dict[str, str]) -> List[str]:
    """
    Given the mapping from entrypoint name to callable, return the relevant
    console script specs.
    """
    console = console.copy()
    scripts_to_generate = []
    pip_script = console.pop('pip', None)
    if pip_script:
        if 'ENSUREPIP_OPTIONS' not in os.environ:
            scripts_to_generate.append('pip = ' + pip_script)
        if os.environ.get('ENSUREPIP_OPTIONS', '') != 'altinstall':
            scripts_to_generate.append(f'pip{sys.version_info[0]} = {pip_script}')
        scripts_to_generate.append(f'pip{get_major_minor_version()} = {pip_script}')
        pip_ep = [k for k in console if re.match('pip(\\d+(\\.\\d+)?)?$', k)]
        for k in pip_ep:
            del console[k]
    easy_install_script = console.pop('easy_install', None)
    if easy_install_script:
        if 'ENSUREPIP_OPTIONS' not in os.environ:
            scripts_to_generate.append('easy_install = ' + easy_install_script)
        scripts_to_generate.append(f'easy_install-{get_major_minor_version()} = {easy_install_script}')
        easy_install_ep = [k for k in console if re.match('easy_install(-\\d+\\.\\d+)?$', k)]
        for k in easy_install_ep:
            del console[k]
    scripts_to_generate.extend(starmap('{} = {}'.format, console.items()))
    return scripts_to_generate