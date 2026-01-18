import unittest
def test___str__w_candidate_no_implementation(self):
    dni = self._makeOne('some_function', '<IFoo>', 'candidate')
    self.assertEqual(str(dni), "The object 'candidate' has failed to implement interface <IFoo>: The contract of 'aMethod' is violated because I said so.")