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
def test_concurrent_deletion(self):
    self.st1._load_from_string(b'foo=bar')
    self.st1.save()
    s1 = self.get_stack(self.st1)
    s2 = self.get_stack(self.st2)
    s1.remove('foo')
    s2.remove('foo')
    s1.store.save_changes()
    self.assertLength(0, self.warnings)
    s2.store.save_changes()
    self.assertLength(1, self.warnings)
    warning = self.warnings[0]
    self.assertStartsWith(warning, 'Option foo in section None')
    self.assertEndsWith(warning, 'was changed from bar to <CREATED>. The <DELETED> value will be saved.')