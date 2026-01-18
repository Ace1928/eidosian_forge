from io import BytesIO
import logging
import os
import re
import struct
import sys
import time
from zipfile import ZipInfo
from .compat import sysconfig, detect_encoding, ZipFile
from .resources import finder
from .util import (FileOperator, get_export_entry, convert_path,
import re
import sys
from %(module)s import %(import_name)s
def enquote_executable(executable):
    if ' ' in executable:
        if executable.startswith('/usr/bin/env '):
            env, _executable = executable.split(' ', 1)
            if ' ' in _executable and (not _executable.startswith('"')):
                executable = '%s "%s"' % (env, _executable)
        elif not executable.startswith('"'):
            executable = '"%s"' % executable
    return executable