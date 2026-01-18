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
def test_last_speaker_wins(self):
    c1 = self.stack
    c2 = self.get_stack(self)
    c1.set('one', 'c1')
    c2.set('one', 'c2')
    self.assertEqual('c2', c2.get('one'))
    self.assertEqual('c1', c1.get('one'))
    c1.set('two', 'done')
    self.assertEqual('c2', c1.get('one'))