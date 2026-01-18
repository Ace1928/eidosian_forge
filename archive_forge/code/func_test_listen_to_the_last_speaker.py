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
def test_listen_to_the_last_speaker(self):
    c1 = self.stack
    c2 = self.get_stack(self)
    c1.set('one', 'ONE')
    c2.set('two', 'TWO')
    self.assertEqual('ONE', c1.get('one'))
    self.assertEqual('TWO', c2.get('two'))
    self.assertEqual('ONE', c2.get('one'))