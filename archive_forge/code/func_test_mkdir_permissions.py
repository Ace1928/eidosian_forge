import os
import stat
import sys
from io import BytesIO
from .. import errors, osutils, pyutils, tests
from .. import transport as _mod_transport
from .. import urlutils
from ..errors import ConnectionError, PathError, TransportNotPossible
from ..osutils import getcwd
from ..transport import (ConnectedTransport, FileExists, NoSuchFile, Transport,
from ..transport.memory import MemoryTransport
from ..transport.remote import RemoteTransport
from . import TestNotApplicable, TestSkipped, multiply_tests, test_server
from .test_transport import TestTransportImplementation
def test_mkdir_permissions(self):
    t = self.get_transport()
    if t.is_readonly():
        return
    if not t._can_roundtrip_unix_modebits():
        return
    t.mkdir('dmode755', mode=493)
    self.assertTransportMode(t, 'dmode755', 493)
    t.mkdir('dmode555', mode=365)
    self.assertTransportMode(t, 'dmode555', 365)
    t.mkdir('dmode777', mode=511)
    self.assertTransportMode(t, 'dmode777', 511)
    t.mkdir('dmode700', mode=448)
    self.assertTransportMode(t, 'dmode700', 448)
    t.mkdir('mdmode755', mode=493)
    self.assertTransportMode(t, 'mdmode755', 493)
    umask = osutils.get_umask()
    t.mkdir('dnomode', mode=None)
    self.assertTransportMode(t, 'dnomode', 511 & ~umask)