import unittest
def test_list_ISequence(self):
    from zope.interface.common.sequence import ISequence
    self._callFUT(ISequence, list, tentative=True)