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
def test_load_hook(self):
    store = self.get_store(self)
    if self.store_id in ('branch', 'remote_branch'):
        self.addCleanup(store.branch.lock_write().unlock)
    section = store.get_mutable_section('baz')
    section.set('foo', 'bar')
    store.save()
    store = self.get_store(self)
    calls = []

    def hook(*args):
        calls.append(args)
    config.ConfigHooks.install_named_hook('load', hook, None)
    self.assertLength(0, calls)
    store.load()
    self.assertLength(1, calls)
    self.assertEqual((store,), calls[0])