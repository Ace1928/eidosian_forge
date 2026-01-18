from __future__ import absolute_import
import socket
from urllib3.contrib import _appengine_environ
from urllib3.exceptions import LocationParseError
from urllib3.packages import six
from .wait import NoWayToWaitForSocketError, wait_for_read
def is_connection_dropped(conn):
    """
    Returns True if the connection is dropped and should be closed.

    :param conn:
        :class:`http.client.HTTPConnection` object.

    Note: For platforms like AppEngine, this will always return ``False`` to
    let the platform handle connection recycling transparently for us.
    """
    sock = getattr(conn, 'sock', False)
    if sock is False:
        return False
    if sock is None:
        return True
    try:
        return wait_for_read(sock, timeout=0.0)
    except NoWayToWaitForSocketError:
        return False