import unittest
from ctypes import *
def test_gh99275(self):

    class BrokenStructure(Structure):

        def __init_subclass__(cls, **kwargs):
            cls._fields_ = []
    with self.assertRaisesRegex(TypeError, 'ctypes state is not initialized'):

        class Subclass(BrokenStructure):
            ...