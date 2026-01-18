from __future__ import absolute_import, print_function
import cython
from .. import __version__
import collections
import contextlib
import hashlib
import os
import shutil
import subprocess
import re, sys, time
from glob import iglob
from io import open as io_open
from os.path import relpath as _relpath
import zipfile
from .. import Utils
from ..Utils import (cached_function, cached_method, path_exists,
from ..Compiler import Errors
from ..Compiler.Main import Context
from ..Compiler.Options import (CompilationOptions, default_options,
@cached_method
def included_files(self, filename):
    all = set()
    for include in self.parse_dependencies(filename)[1]:
        include_path = join_path(os.path.dirname(filename), include)
        if not path_exists(include_path):
            include_path = self.context.find_include_file(include, source_file_path=filename)
        if include_path:
            if '.' + os.path.sep in include_path:
                include_path = os.path.normpath(include_path)
            all.add(include_path)
            all.update(self.included_files(include_path))
        elif not self.quiet:
            print(u"Unable to locate '%s' referenced from '%s'" % (filename, include))
    return all