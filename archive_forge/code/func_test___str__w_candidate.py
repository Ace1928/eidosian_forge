import unittest
def test___str__w_candidate(self):
    dni = self._makeOne('candidate')
    self.assertEqual(str(dni), "The object 'candidate' has failed to implement interface zope.interface.tests.test_exceptions.IDummy: The 'missing' attribute was not provided.")