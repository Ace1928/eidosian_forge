import subprocess
import doctest
import os
import sys
import shutil
import re
import cgi
import rfc822
from io import StringIO
from paste.util import PySourceColor
def convert_docstring_string(data):
    if data.startswith('\n'):
        data = data[1:]
    lines = data.splitlines()
    new_lines = []
    for line in lines:
        if line.rstrip() == '.':
            new_lines.append('')
        else:
            new_lines.append(line)
    data = '\n'.join(new_lines) + '\n'
    return data