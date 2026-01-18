import sys
import unittest
import sys
def test_meta_of_class(self):

    class Metameta(type):
        pass

    class Meta(type, metaclass=Metameta):
        pass
    self.assertEqual(self._callFUT((Meta, type)), Metameta)