import sys
import os
import shutil
import io
import re
import textwrap
from os.path import relpath
from errno import EEXIST
import traceback
def out_of_date(original, derived):
    """
    Returns True if derivative is out-of-date wrt original,
    both of which are full file paths.
    """
    return not os.path.exists(derived) or (os.path.exists(original) and os.stat(derived).st_mtime < os.stat(original).st_mtime)