import os
import platform
import socket
import stat
import six
from oauthlib import oauth1
from six.moves.urllib.parse import parse_qs, urlencode
from lazr.restfulclient.authorize import HttpAuthorizer
from lazr.restfulclient.errors import CredentialsFileError
@property
def consumer_key(self):
    """The system-wide OAuth consumer key for this computer.

        This key identifies the platform and the computer's
        hostname. It does not identify the active user.
        """
    try:
        import distro
        distname = distro.name()
    except Exception:
        distname = ''
    if distname == '':
        distname = platform.system()
    return self.KEY_FORMAT % (distname, socket.gethostname())