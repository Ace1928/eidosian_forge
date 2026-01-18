import codecs
from collections import deque
import contextlib
import csv
from glob import iglob as std_iglob
import io
import json
import logging
import os
import py_compile
import re
import socket
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
from . import DistlibException
from .compat import (string_types, text_type, shutil, raw_input, StringIO,
def get_export_entry(specification):
    m = ENTRY_RE.search(specification)
    if not m:
        result = None
        if '[' in specification or ']' in specification:
            raise DistlibException("Invalid specification '%s'" % specification)
    else:
        d = m.groupdict()
        name = d['name']
        path = d['callable']
        colons = path.count(':')
        if colons == 0:
            prefix, suffix = (path, None)
        else:
            if colons != 1:
                raise DistlibException("Invalid specification '%s'" % specification)
            prefix, suffix = path.split(':')
        flags = d['flags']
        if flags is None:
            if '[' in specification or ']' in specification:
                raise DistlibException("Invalid specification '%s'" % specification)
            flags = []
        else:
            flags = [f.strip() for f in flags.split(',')]
        result = ExportEntry(name, prefix, suffix, flags)
    return result