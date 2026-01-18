import unittest
def test_complex_diamond_compare_legacy_argument(self):
    from zope.interface import Interface
    A = self._make_complex_diamond(Interface)
    computed_A_iro = self._callFUT(A, log_changed_ro=True)
    self.assertEqual(tuple(computed_A_iro), A.__iro__)
    self._check_handler_complex_diamond()