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
def test_clone_url_unquote_unreserved(self):
    """Base URL of a cloned branch needs unreserved characters unquoted

        Cloned transports should be prefix comparable for things like the
        isolation checking of tests, see lp:842223 for more.
        """
    t1 = self.get_transport()
    needlessly_escaped_dir = '%2D%2E%30%39%41%5A%5F%61%7A%7E/'
    self.build_tree([needlessly_escaped_dir], transport=t1)
    t2 = t1.clone(needlessly_escaped_dir)
    self.assertEqual(t1.base + '-.09AZ_az~/', t2.base)