import os
import sys
from breezy import osutils, tests, urlutils
from breezy.tests import EncodingAdapter
from breezy.tests.scenarios import load_tests_apply_scenarios
def run_bzr_decode(self, args, encoding=None, fail=False, retcode=None, working_dir=None):
    """Run brz and decode the output into a particular encoding.

        Returns a string containing the stdout output from bzr.

        :param fail: If true, the operation is expected to fail with
            a UnicodeError.
        """
    if encoding is None:
        encoding = osutils.get_user_encoding()
    try:
        out = self.run_bzr_raw(args, encoding=encoding, retcode=retcode, working_dir=working_dir)[0]
        return out.decode(encoding)
    except UnicodeError as e:
        if not fail:
            raise
    else:
        if fail:
            self.fail('Expected UnicodeError not raised')