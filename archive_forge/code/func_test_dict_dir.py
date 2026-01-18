import unittest
from IPython.utils import wildcard
def test_dict_dir(self):

    class A(object):

        def __init__(self):
            self.a = 1
            self.b = 2

        def __getattribute__(self, name):
            if name == 'a':
                raise AttributeError
            return object.__getattribute__(self, name)
    a = A()
    adict = wildcard.dict_dir(a)
    assert 'a' not in adict
    self.assertEqual(adict['b'], 2)