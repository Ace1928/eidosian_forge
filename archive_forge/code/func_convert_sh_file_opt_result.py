from __future__ import unicode_literals
import os.path as op
from send2trash.compat import text_type
from send2trash.util import preprocess_paths
from ctypes import (
from ctypes.wintypes import HWND, UINT, LPCWSTR, BOOL
def convert_sh_file_opt_result(result):
    results = {113: 80, 114: 87, 115: 87, 116: 87, 117: 1223, 118: 87, 120: 5, 121: 111, 122: 87, 124: 161, 125: 87, 126: 183, 128: 183, 129: 111, 130: 19, 131: 19, 132: 1785, 133: 223, 134: 19, 135: 19, 136: 1785, 183: 111, 1026: 161, 65536: 29, 65652: 87}
    return results.get(result, result)