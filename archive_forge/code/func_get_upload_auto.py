import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def get_upload_auto(self):
    b = controldir.ControlDir.open(self.tree.basedir).open_branch()
    return b.get_config_stack().get('upload_auto')