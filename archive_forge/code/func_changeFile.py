import os
import sys
from breezy import osutils
from breezy.tests import (TestCaseWithTransport, TestNotApplicable,
from breezy.workingtree import WorkingTree
def changeFile(self, file_name):
    with open(file_name, 'ab') as f:
        f.write(b'\nsome other new content!')