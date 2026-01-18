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
def test_get_default_list_from_env(self):
    self.register_list_option('foo', default_from_env=['FOO'])
    self.overrideEnv('FOO', '')
    conf = self.get_conf(b'')
    self.assertEqual([], conf.get('foo'))