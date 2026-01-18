import fnmatch
import locale
import os
import re
import stat
import subprocess
import sys
import textwrap
import types
import warnings
from xml.etree import ElementTree
def import_from_stdlib(module):
    """
    When python is run from within the nltk/ directory tree, the
    current directory is included at the beginning of the search path.
    Unfortunately, that means that modules within nltk can sometimes
    shadow standard library modules.  As an example, the stdlib
    'inspect' module will attempt to import the stdlib 'tokenize'
    module, but will instead end up importing NLTK's 'tokenize' module
    instead (causing the import to fail).
    """
    old_path = sys.path
    sys.path = [d for d in sys.path if d not in ('', '.')]
    m = __import__(module)
    sys.path = old_path
    return m