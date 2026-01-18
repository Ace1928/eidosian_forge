import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def assertUpPathDoesNotExist(self, path, base=upload_dir):
    self.assertPathDoesNotExist(osutils.pathjoin(base, path))