import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def assertUpPathModeEqual(self, path, expected_mode, base=upload_dir):
    full_path = osutils.pathjoin(base, path)
    st = os.stat(full_path)
    mode = st.st_mode & 511
    if expected_mode == mode:
        return
    raise AssertionError('For path %s, mode is %s not %s' % (full_path, oct(mode), oct(expected_mode)))