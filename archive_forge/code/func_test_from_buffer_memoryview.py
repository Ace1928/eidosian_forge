from ctypes import *
import array
import gc
import unittest
def test_from_buffer_memoryview(self):
    a = [c_char.from_buffer(memoryview(bytearray(b'a')))]
    a.append(a)
    del a
    gc.collect()