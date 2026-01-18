import os
import sys
from breezy import urlutils
from breezy.branch import Branch
from breezy.controldir import ControlDir
from breezy.tests import TestCaseWithTransport, TestSkipped
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
from breezy.workingtree import WorkingTree
def check_mode_r(test, base, file_mode, dir_mode, include_base=True):
    """Check that all permissions match

    :param test: The TestCase being run
    :param base: The path to the root directory to check
    :param file_mode: The mode for all files
    :param dir_mode: The mode for all directories
    :param include_base: If false, only check the subdirectories
    """
    t = test.get_transport()
    if include_base:
        test.assertTransportMode(t, base, dir_mode)
    for root, dirs, files in os.walk(base):
        for d in dirs:
            p = '/'.join([urlutils.quote(x) for x in root.split('/\\') + [d]])
            test.assertTransportMode(t, p, dir_mode)
        for f in files:
            p = os.path.join(root, f)
            p = '/'.join([urlutils.quote(x) for x in root.split('/\\') + [f]])
            test.assertTransportMode(t, p, file_mode)