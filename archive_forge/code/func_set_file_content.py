import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def set_file_content(self, path, content, base=branch_dir):
    with open(osutils.pathjoin(base, path), 'wb') as f:
        f.write(content)