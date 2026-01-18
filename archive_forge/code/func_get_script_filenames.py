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
def get_script_filenames(self, name):
    result = set()
    if '' in self.variants:
        result.add(name)
    if 'X' in self.variants:
        result.add('%s%s' % (name, self.version_info[0]))
    if 'X.Y' in self.variants:
        result.add('%s%s%s.%s' % (name, self.variant_separator, self.version_info[0], self.version_info[1]))
    return result