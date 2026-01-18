from __future__ import (absolute_import, division, print_function)
import json
import platform
import io
import os
def read_utf8_file(path, encoding='utf-8'):
    if not os.access(path, os.R_OK):
        return None
    with io.open(path, 'r', encoding=encoding) as fd:
        content = fd.read()
    return content