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
def fix_script(path: str) -> bool:
    """Replace #!python with #!/path/to/python
    Return True if file was changed.
    """
    assert os.path.isfile(path)
    with open(path, 'rb') as script:
        firstline = script.readline()
        if not firstline.startswith(b'#!python'):
            return False
        exename = sys.executable.encode(sys.getfilesystemencoding())
        firstline = b'#!' + exename + os.linesep.encode('ascii')
        rest = script.read()
    with open(path, 'wb') as script:
        script.write(firstline)
        script.write(rest)
    return True