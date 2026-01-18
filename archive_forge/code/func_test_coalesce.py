import yaql.tests
def test_coalesce(self):
    self.assertEqual(2, self.eval('coalesce($, 2)', data=None))
    self.assertEqual(1, self.eval('coalesce($, 2)', data=1))
    self.assertEqual(2, self.eval('coalesce($, $, 2)', data=None))
    self.assertIsNone(self.eval('coalesce($)', data=None))