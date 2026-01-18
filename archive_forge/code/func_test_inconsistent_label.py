import unittest
def test_inconsistent_label(self):
    comp = self._makeOne()
    self.assertEqual('no', comp._inconsistent_label)
    comp.c3.direct_inconsistency = True
    self.assertEqual('direct', comp._inconsistent_label)
    comp.c3.bases_had_inconsistency = True
    self.assertEqual('direct+bases', comp._inconsistent_label)
    comp.c3.direct_inconsistency = False
    self.assertEqual('bases', comp._inconsistent_label)