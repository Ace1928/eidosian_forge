import os
import sys
import threading
from io import BytesIO
from textwrap import dedent
import configobj
from testtools import matchers
from .. import (bedding, branch, config, controldir, diff, errors, lock,
from .. import registry as _mod_registry
from .. import tests, trace
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..bzr import remote
from ..transport import remote as transport_remote
from . import features, scenarios, test_server
def test_ini_config_ownership(self):
    """Ensure that chown is happening during _write_config_file"""
    self.requireFeature(features.chown_feature)
    self.overrideAttr(os, 'chown', self._dummy_chown)
    self.path = self.uid = self.gid = None
    conf = config.IniBasedConfig(file_name='./foo.conf')
    conf._write_config_file()
    self.assertEqual(self.path, './foo.conf')
    self.assertTrue(isinstance(self.uid, int))
    self.assertTrue(isinstance(self.gid, int))