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
def test__reuse_for(self):
    t = self.get_transport()
    if not isinstance(t, ConnectedTransport):
        raise TestSkipped('not a connected transport')

    def new_url(scheme=None, user=None, password=None, host=None, port=None, path=None):
        """Build a new url from t.base changing only parts of it.

            Only the parameters different from None will be changed.
            """
        if scheme is None:
            scheme = t._parsed_url.scheme
        if user is None:
            user = t._parsed_url.user
        if password is None:
            password = t._parsed_url.password
        if user is None:
            user = t._parsed_url.user
        if host is None:
            host = t._parsed_url.host
        if port is None:
            port = t._parsed_url.port
        if path is None:
            path = t._parsed_url.path
        return str(urlutils.URL(scheme, user, password, host, port, path))
    if t._parsed_url.scheme == 'ftp':
        scheme = 'sftp'
    else:
        scheme = 'ftp'
    self.assertIsNot(t, t._reuse_for(new_url(scheme=scheme)))
    if t._parsed_url.user == 'me':
        user = 'you'
    else:
        user = 'me'
    self.assertIsNot(t, t._reuse_for(new_url(user=user)))
    self.assertIs(t, t._reuse_for(new_url(password='from space')))
    self.assertIsNot(t, t._reuse_for(new_url(host=t._parsed_url.host + 'bar')))
    if t._parsed_url.port == 1234:
        port = 4321
    else:
        port = 1234
    self.assertIsNot(t, t._reuse_for(new_url(port=port)))
    self.assertIs(None, t._reuse_for('/valid_but_not_existing'))