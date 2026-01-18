import os
import abc
import sys
from Cryptodome.Util.py3compat import byte_string
from Cryptodome.Util._file_system import pycryptodome_filename
def c_uint8_ptr(data):
    if byte_string(data) or isinstance(data, _Array):
        return data
    elif isinstance(data, _buffer_type):
        obj = _py_object(data)
        buf = _Py_buffer()
        _PyObject_GetBuffer(obj, byref(buf), _PyBUF_SIMPLE)
        try:
            buffer_type = ctypes.c_ubyte * buf.len
            return buffer_type.from_address(buf.buf)
        finally:
            _PyBuffer_Release(byref(buf))
    else:
        raise TypeError('Object type %s cannot be passed to C code' % type(data))