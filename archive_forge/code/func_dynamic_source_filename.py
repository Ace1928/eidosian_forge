from __future__ import absolute_import
import re
import os.path
import sys
from collections import defaultdict
from coverage.plugin import CoveragePlugin, FileTracer, FileReporter  # requires coverage.py 4.0+
from coverage.files import canonical_filename
from .Utils import find_root_package_dir, is_package_dir, is_cython_generated_file, open_source_file
from . import __version__
def dynamic_source_filename(self, filename, frame):
    """
        Determine source file path.  Called by the function call tracer.
        """
    source_file = frame.f_code.co_filename
    try:
        return self._file_path_map[source_file]
    except KeyError:
        pass
    abs_path = _find_dep_file_path(filename, source_file)
    if self.py_file and source_file[-3:].lower() == '.py':
        self._file_path_map[source_file] = self.py_file
        return self.py_file
    assert self._c_files_map is not None
    if abs_path not in self._c_files_map:
        self._c_files_map[abs_path] = (self.c_file, source_file, None)
    self._file_path_map[source_file] = abs_path
    return abs_path