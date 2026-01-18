import re
import sys
import types
import fnmatch
from os.path import basename
from pygments.lexers._mapping import LEXERS
from pygments.modeline import get_filetype_from_buffer
from pygments.plugin import find_plugin_lexers
from pygments.util import ClassNotFound, itervalues, guess_decode
def get_rating(info):
    cls, filename = info
    bonus = '*' not in filename and 0.5 or 0
    if code:
        return (cls.analyse_text(code) + bonus, cls.__name__)
    return (cls.priority + bonus, cls.__name__)