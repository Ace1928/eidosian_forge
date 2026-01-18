import os
import shutil
import stat
import sys
from ...controldir import ControlDir
from .. import KnownFailure, TestCaseWithTransport, TestSkipped
def testRunScriptMergePoint(self):
    """Make a test script and run it."""
    if sys.platform == 'win32':
        raise TestSkipped('Unable to run shell script on windows')
    with open('test_script', 'w') as test_script:
        test_script.write("#!/bin/sh\ngrep -q '^two' test_file_append\n")
    os.chmod('test_script', stat.S_IRWXU)
    self.run_bzr(['bisect', 'start'])
    self.run_bzr(['bisect', 'yes'])
    self.run_bzr(['bisect', 'no', '-r', '1'])
    self.run_bzr(['bisect', 'run', './test_script'])
    try:
        self.assertRevno(2)
    except AssertionError:
        raise KnownFailure('bisect does not drill down into merge commits: https://bugs.launchpad.net/bzr-bisect/+bug/539937')