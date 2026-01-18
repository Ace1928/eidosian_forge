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
def set_mask_key(self, func):
    """
        set function to create musk key. You can customize mask key generator.
        Mainly, this is for testing purpose.

        func: callable object. the func takes 1 argument as integer.
              The argument means length of mask key.
              This func must return string(byte array),
              which length is argument specified.
        """
    self.get_mask_key = func