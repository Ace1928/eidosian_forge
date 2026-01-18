import os
import re
import sys
import breezy
from breezy import osutils
from breezy.branch import Branch
from breezy.errors import CommandError
from breezy.tests import TestCaseWithTransport
from breezy.tests.http_utils import TestCaseWithWebserver
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
from breezy.workingtree import WorkingTree
def listdir_sorted(dir):
    L = sorted(os.listdir(dir))
    return L