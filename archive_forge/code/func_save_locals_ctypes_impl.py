import sys
def save_locals_ctypes_impl(frame):
    locals_to_fast(ctypes.py_object(frame), ctypes.c_int(0))