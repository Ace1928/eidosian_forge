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
def append_to_file(path, version, data):
    data = convert_docstring_string(data)
    f = open(path, 'a')
    f.write(data)
    f.close()
    if path.endswith('.py'):
        pyc_file = path + 'c'
        if os.path.exists(pyc_file):
            os.unlink(pyc_file)
    show_file(path, version, description='added to %s' % path, data=data)