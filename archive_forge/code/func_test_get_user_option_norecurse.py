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
def test_get_user_option_norecurse(self):
    self.get_branch_config('http://www.example.com')
    self.assertEqual('norecurse', self.my_config.get_user_option('norecurse_option'))
    self.get_branch_config('http://www.example.com/dir')
    self.assertEqual(None, self.my_config.get_user_option('norecurse_option'))
    self.get_branch_config('http://www.example.com/norecurse')
    self.assertEqual('norecurse', self.my_config.get_user_option('normal_option'))
    self.get_branch_config('http://www.example.com/norecurse/subdir')
    self.assertEqual('normal', self.my_config.get_user_option('normal_option'))