import os
from breezy import tests
from breezy.tests import script
def test_failing_script(self):
    self.build_tree_contents([('script', b'\n$ echo hello foo\nhello bar\n')])
    out, err = self.run_bzr(['test-script', 'script'], retcode=1)
    out_lines = out.splitlines()
    self.assertStartsWith(out_lines[-3], 'Ran 1 test in ')
    self.assertEqual('FAILED (failures=1)', out_lines[-1])
    self.assertEqual('', err)