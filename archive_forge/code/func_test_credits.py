from ...tests import TestCaseWithTransport
def test_credits(self):
    out, err = self.run_bzr('credits')
    self.assertEqual(out, 'Code:\nFero <fero@example.com>\n\n')