import os
from breezy import tests
from breezy.tests import features
from breezy.transport import memory
def outputs_nothing(cmdline):
    out, err = self.run_bzr(cmdline)
    header, body, footer = self._parse_test_list(out.splitlines())
    num_tests = len(body)
    self.assertLength(0, header)
    self.assertLength(0, footer)
    self.assertEqual('', err)