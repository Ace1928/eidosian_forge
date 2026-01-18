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
def test_get_user_option_appendpath(self):
    self.get_branch_config('http://www.example.com')
    self.assertEqual('append', self.my_config.get_user_option('appendpath_option'))
    self.get_branch_config('http://www.example.com/a/b/c')
    self.assertEqual('append/a/b/c', self.my_config.get_user_option('appendpath_option'))
    self.get_branch_config('http://www.example.com/dir/a/b/c')
    self.assertEqual('normal', self.my_config.get_user_option('appendpath_option'))