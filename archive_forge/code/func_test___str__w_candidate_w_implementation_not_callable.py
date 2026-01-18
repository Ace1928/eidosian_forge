import unittest
def test___str__w_candidate_w_implementation_not_callable(self):
    self.message = 'implementation is not callable'
    dni = self._makeOne(42, '<IFoo>', 'candidate')
    self.assertEqual(str(dni), "The object 'candidate' has failed to implement interface <IFoo>: The contract of 'aMethod' is violated because '42' is not callable.")