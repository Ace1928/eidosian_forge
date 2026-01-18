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
def assertRemoveHook(self, conf, name, section_name=None):
    calls = []

    def hook(*args):
        calls.append(args)
    config.OldConfigHooks.install_named_hook('remove', hook, None)
    self.addCleanup(config.OldConfigHooks.uninstall_named_hook, 'remove', None)
    self.assertLength(0, calls)
    conf.remove_user_option(name, section_name)
    self.assertLength(1, calls)
    self.assertEqual((name,), calls[0][1:])