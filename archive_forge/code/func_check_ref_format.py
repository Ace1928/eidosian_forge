import re
import sys
import time
from email.utils import parseaddr
import breezy.branch
import breezy.revision
from ... import (builtins, errors, lazy_import, lru_cache, osutils, progress,
from ... import transport as _mod_transport
from . import helpers, marks_file
from fastimport import commands
def check_ref_format(refname):
    """Check if a refname is correctly formatted.

    Implements all the same rules as git-check-ref-format[1].

    [1] http://www.kernel.org/pub/software/scm/git/docs/git-check-ref-format.html

    :param refname: The refname to check
    :return: True if refname is valid, False otherwise
    """
    if b'/.' in refname or refname.startswith(b'.'):
        return False
    if b'/' not in refname:
        return False
    if b'..' in refname:
        return False
    for i in range(len(refname)):
        if ord(refname[i:i + 1]) < 32 or refname[i] in b'\x7f ~^:?*[':
            return False
    if refname[-1] in b'/.':
        return False
    if refname.endswith(b'.lock'):
        return False
    if b'@{' in refname:
        return False
    if b'\\' in refname:
        return False
    return True