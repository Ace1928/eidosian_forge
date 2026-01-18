import sys
import unittest
import platform
import pygame
def test_GetView_array_struct(self):
    from pygame.bufferproxy import BufferProxy

    class Exporter(self.ExporterBase):

        def __init__(self, shape, typechar, itemsize):
            super().__init__(shape, typechar, itemsize)
            self.view = BufferProxy(self.__dict__)

        def get__array_struct__(self):
            return self.view.__array_struct__
        __array_struct__ = property(get__array_struct__)
        __array_interface__ = property(lambda self: None)
    _shape = [2, 3, 5, 7, 11]
    for ndim in range(1, len(_shape)):
        o = Exporter(_shape[0:ndim], 'i', 2)
        v = BufferProxy(o)
        self.assertSame(v, o)
    ndim = 2
    shape = _shape[0:ndim]
    for typechar in ('i', 'u'):
        for itemsize in (1, 2, 4, 8):
            o = Exporter(shape, typechar, itemsize)
            v = BufferProxy(o)
            self.assertSame(v, o)
    for itemsize in (4, 8):
        o = Exporter(shape, 'f', itemsize)
        v = BufferProxy(o)
        self.assertSame(v, o)
    try:
        from sys import getrefcount
    except ImportError:
        pass
    else:
        o = Exporter(shape, typechar, itemsize)
        self.assertEqual(getrefcount(o.__array_struct__), 1)