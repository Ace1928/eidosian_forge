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
def test_abspath_url_unquote_unreserved(self):
    """URLs from abspath should have unreserved characters unquoted

        Need consistent quoting notably for tildes, see lp:842223 for more.
        """
    t = self.get_transport()
    needlessly_escaped_dir = '%2D%2E%30%39%41%5A%5F%61%7A%7E/'
    self.assertEqual(t.base + '-.09AZ_az~', t.abspath(needlessly_escaped_dir))