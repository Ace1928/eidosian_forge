import os
import socket
import typing
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL._util import (
from OpenSSL._util import (
from OpenSSL._util import (
from OpenSSL._util import (
from OpenSSL._util import (
from OpenSSL._util import (
from OpenSSL._util import (
from OpenSSL.crypto import (
def get_selected_srtp_profile(self):
    """
        Get the SRTP protocol which was negotiated.

        :returns: A bytestring of the SRTP profile name. If no profile has been
            negotiated yet, returns an empty bytestring.
        """
    profile = _lib.SSL_get_selected_srtp_profile(self._ssl)
    if not profile:
        return b''
    return _ffi.string(profile.name)