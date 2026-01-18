import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def do_incremental_upload(self, *args, **kwargs):
    upload = self._get_cmd_upload()
    up_url = self.get_url(self.upload_dir)
    if kwargs.get('directory', None) is None:
        kwargs['directory'] = self.branch_dir
    kwargs['quiet'] = True
    upload.run(up_url, *args, **kwargs)