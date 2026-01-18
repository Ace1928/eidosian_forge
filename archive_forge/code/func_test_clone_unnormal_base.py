import doctest
import errno
import os
import socket
import subprocess
import sys
import threading
import time
from io import BytesIO
from typing import Optional, Type
from testtools.matchers import DocTestMatches
import breezy
from ... import controldir, debug, errors, osutils, tests
from ... import transport as _mod_transport
from ... import urlutils
from ...tests import features, test_server
from ...transport import local, memory, remote, ssh
from ...transport.http import urllib
from .. import bzrdir
from ..remote import UnknownErrorFromSmartServer
from ..smart import client, medium, message, protocol
from ..smart import request as _mod_request
from ..smart import server as _mod_server
from ..smart import vfs
from . import test_smart
def test_clone_unnormal_base(self):
    base_transport = remote.RemoteHTTPTransport('bzr+http://host/%7Ea/b')
    new_transport = base_transport.clone('c')
    self.assertEqual(base_transport.base + 'c/', new_transport.base)
    self.assertEqual(b'c/', new_transport._client.remote_path_from_transport(new_transport))