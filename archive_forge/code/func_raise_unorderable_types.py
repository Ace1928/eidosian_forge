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
def raise_unorderable_types(ordering, a, b):
    raise TypeError('unorderable types: %s() %s %s()' % (type(a).__name__, ordering, type(b).__name__))