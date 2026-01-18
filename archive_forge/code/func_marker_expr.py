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
def marker_expr(remaining):
    if remaining and remaining[0] == '(':
        result, remaining = marker(remaining[1:].lstrip())
        if remaining[0] != ')':
            raise SyntaxError('unterminated parenthesis: %s' % remaining)
        remaining = remaining[1:].lstrip()
    else:
        lhs, remaining = marker_var(remaining)
        while remaining:
            m = MARKER_OP.match(remaining)
            if not m:
                break
            op = m.groups()[0]
            remaining = remaining[m.end():]
            rhs, remaining = marker_var(remaining)
            lhs = {'op': op, 'lhs': lhs, 'rhs': rhs}
        result = lhs
    return (result, remaining)