import os
from breezy import tests
from breezy.tests import features
from breezy.transport import memory
def test_list_only(self):

    def selftest(*args, **kwargs):
        """Capture the arguments selftest was run with."""
        return True

    def outputs_nothing(cmdline):
        out, err = self.run_bzr(cmdline)
        header, body, footer = self._parse_test_list(out.splitlines())
        num_tests = len(body)
        self.assertLength(0, header)
        self.assertLength(0, footer)
        self.assertEqual('', err)
    original_selftest = tests.selftest
    tests.selftest = selftest
    try:
        outputs_nothing('selftest --list-only')
        outputs_nothing('selftest --list-only selftest')
        outputs_nothing(['selftest', '--list-only', '--exclude', 'selftest'])
    finally:
        tests.selftest = original_selftest