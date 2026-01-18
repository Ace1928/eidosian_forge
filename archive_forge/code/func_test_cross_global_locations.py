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
def test_cross_global_locations(self):
    l_store = config.LocationStore()
    l_store._load_from_string(b'\n[/branch]\nlfoo = loc-foo\nlbar = {gbar}\n')
    l_store.save()
    g_store = config.GlobalStore()
    g_store._load_from_string(b'\n[DEFAULT]\ngfoo = {lfoo}\ngbar = glob-bar\n')
    g_store.save()
    stack = config.LocationStack('/branch')
    self.assertEqual('glob-bar', stack.get('lbar', expand=True))
    self.assertEqual('loc-foo', stack.get('gfoo', expand=True))