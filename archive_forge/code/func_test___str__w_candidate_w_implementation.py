import unittest
def test___str__w_candidate_w_implementation(self):
    self.message = 'implementation is wonky'
    dni = self._makeOne(broken_function, '<IFoo>', 'candidate')
    self.assertEqual(str(dni), "The object 'candidate' has failed to implement interface <IFoo>: The contract of 'aMethod' is violated because 'broken_function()' is wonky.")