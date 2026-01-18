from __future__ import print_function
import socket
import struct
import threading
import time
import six
from ._abnf import *
from ._exceptions import *
from ._handshake import *
from ._http import *
from ._logging import *
from ._socket import *
from ._ssl_compat import *
from ._utils import *
def getstatus(self):
    """
        get handshake status
        """
    if self.handshake_response:
        return self.handshake_response.status
    else:
        return None