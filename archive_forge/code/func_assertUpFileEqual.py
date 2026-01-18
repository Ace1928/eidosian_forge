import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def assertUpFileEqual(self, content, path, base=upload_dir):
    self.assertFileEqual(content, osutils.pathjoin(base, path))