from __future__ import unicode_literals
from ctypes import cdll, byref, Structure, c_char, c_char_p
from ctypes.util import find_library
from send2trash.compat import binary_type
from send2trash.util import preprocess_paths
def check_op_result(op_result):
    if op_result:
        msg = GetMacOSStatusCommentString(op_result).decode('utf-8')
        raise OSError(msg)