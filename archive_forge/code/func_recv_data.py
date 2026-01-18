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
def recv_data(self, control_frame=False):
    """
        Receive data with operation code.

        control_frame: a boolean flag indicating whether to return control frame
        data, defaults to False

        return  value: tuple of operation code and string(byte array) value.
        """
    opcode, frame = self.recv_data_frame(control_frame)
    return (opcode, frame.data)