from ctypes import *
import array
import gc
import unittest
def test_from_buffer(self):
    a = array.array('i', range(16))
    x = (c_int * 16).from_buffer(a)
    y = X.from_buffer(a)
    self.assertEqual(y.c_int, a[0])
    self.assertFalse(y.init_called)
    self.assertEqual(x[:], a.tolist())
    a[0], a[-1] = (200, -200)
    self.assertEqual(x[:], a.tolist())
    self.assertRaises(BufferError, a.append, 100)
    self.assertRaises(BufferError, a.pop)
    del x
    del y
    gc.collect()
    gc.collect()
    gc.collect()
    a.append(100)
    a.pop()
    x = (c_int * 16).from_buffer(a)
    self.assertIn(a, [obj.obj if isinstance(obj, memoryview) else obj for obj in x._objects.values()])
    expected = x[:]
    del a
    gc.collect()
    gc.collect()
    gc.collect()
    self.assertEqual(x[:], expected)
    with self.assertRaisesRegex(TypeError, 'not writable'):
        (c_char * 16).from_buffer(b'a' * 16)
    with self.assertRaisesRegex(TypeError, 'not writable'):
        (c_char * 16).from_buffer(memoryview(b'a' * 16))
    with self.assertRaisesRegex(TypeError, 'not C contiguous'):
        (c_char * 16).from_buffer(memoryview(bytearray(b'a' * 16))[::-1])
    msg = 'bytes-like object is required'
    with self.assertRaisesRegex(TypeError, msg):
        (c_char * 16).from_buffer('a' * 16)