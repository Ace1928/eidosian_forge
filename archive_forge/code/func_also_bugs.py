import collections.abc
import datetime
import email.utils
import functools
import logging
import io
import re
import subprocess
import warnings
import chardet
from debian._util import (
from debian.deprecation import function_deprecated_by
import debian.debian_support
import debian.changelog
@property
def also_bugs(self):
    """ list of bug numbers in the package closed by the removal

        The bug numbers are returned as integers.

        Removal of a package implicitly also closes all bugs associated with
        the package.
        """
    if 'also-bugs' not in self:
        return []
    return [int(b) for b in self['also-bugs'].split(' ')]