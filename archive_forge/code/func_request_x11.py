import binascii
import os
import socket
import time
import threading
from functools import wraps
from paramiko import util
from paramiko.common import (
from paramiko.message import Message
from paramiko.ssh_exception import SSHException
from paramiko.file import BufferedFile
from paramiko.buffered_pipe import BufferedPipe, PipeTimeout
from paramiko import pipe
from paramiko.util import ClosingContextManager
@open_only
def request_x11(self, screen_number=0, auth_protocol=None, auth_cookie=None, single_connection=False, handler=None):
    """
        Request an x11 session on this channel.  If the server allows it,
        further x11 requests can be made from the server to the client,
        when an x11 application is run in a shell session.

        From :rfc:`4254`::

            It is RECOMMENDED that the 'x11 authentication cookie' that is
            sent be a fake, random cookie, and that the cookie be checked and
            replaced by the real cookie when a connection request is received.

        If you omit the auth_cookie, a new secure random 128-bit value will be
        generated, used, and returned.  You will need to use this value to
        verify incoming x11 requests and replace them with the actual local
        x11 cookie (which requires some knowledge of the x11 protocol).

        If a handler is passed in, the handler is called from another thread
        whenever a new x11 connection arrives.  The default handler queues up
        incoming x11 connections, which may be retrieved using
        `.Transport.accept`.  The handler's calling signature is::

            handler(channel: Channel, (address: str, port: int))

        :param int screen_number: the x11 screen number (0, 10, etc.)
        :param str auth_protocol:
            the name of the X11 authentication method used; if none is given,
            ``"MIT-MAGIC-COOKIE-1"`` is used
        :param str auth_cookie:
            hexadecimal string containing the x11 auth cookie; if none is
            given, a secure random 128-bit value is generated
        :param bool single_connection:
            if True, only a single x11 connection will be forwarded (by
            default, any number of x11 connections can arrive over this
            session)
        :param handler:
            an optional callable handler to use for incoming X11 connections
        :return: the auth_cookie used
        """
    if auth_protocol is None:
        auth_protocol = 'MIT-MAGIC-COOKIE-1'
    if auth_cookie is None:
        auth_cookie = binascii.hexlify(os.urandom(16))
    m = Message()
    m.add_byte(cMSG_CHANNEL_REQUEST)
    m.add_int(self.remote_chanid)
    m.add_string('x11-req')
    m.add_boolean(True)
    m.add_boolean(single_connection)
    m.add_string(auth_protocol)
    m.add_string(auth_cookie)
    m.add_int(screen_number)
    self._event_pending()
    self.transport._send_user_message(m)
    self._wait_for_event()
    self.transport._set_x11_handler(handler)
    return auth_cookie