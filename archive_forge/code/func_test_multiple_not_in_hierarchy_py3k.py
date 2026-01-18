import sys
import unittest
import sys
def test_multiple_not_in_hierarchy_py3k(self):

    class Meta_A(type):
        pass

    class Meta_B(type):
        pass

    class A(type, metaclass=Meta_A):
        pass

    class B(type, metaclass=Meta_B):
        pass
    self.assertRaises(TypeError, self._callFUT, (A, B))