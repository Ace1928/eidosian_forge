import sys
import os
import shutil
import io
import re
import textwrap
from os.path import relpath
from errno import EEXIST
import traceback
def contains_doctest(text):
    try:
        compile(text, '<string>', 'exec')
        return False
    except SyntaxError:
        pass
    r = re.compile('^\\s*>>>', re.M)
    m = r.search(text)
    return bool(m)