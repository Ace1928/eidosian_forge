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
def create_pipe_context(self, to_server_bytes, transport):
    """Create a SmartServerSocketStreamMedium.

        This differes from create_pipe_medium, in that we initialize the
        request that is sent to the server, and return the BytesIO class that
        will hold the response.
        """
    to_server = BytesIO(to_server_bytes)
    from_server = BytesIO()
    m = self.create_pipe_medium(to_server, from_server, transport)
    return (m, from_server)