import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def test_get_upload_location_unset(self):
    conf = self.get_branch().get_config_stack()
    self.assertEqual(None, conf.get('upload_location'))