import bisect
from _pydevd_bundle.pydevd_constants import NULL, KeyifyList
import pydevd_file_utils
def contains_runtime_line(self, i):
    line_count = self.end_line + self.line
    runtime_end_line = self.runtime_line + line_count
    return self.runtime_line <= i <= runtime_end_line