import bisect
from _pydevd_bundle.pydevd_constants import NULL, KeyifyList
import pydevd_file_utils
def contains_line(self, i):
    return self.line <= i <= self.end_line