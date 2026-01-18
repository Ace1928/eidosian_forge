from ncclient import operations
from ncclient import transport
import socket
import logging
import functools
from ncclient.xml_ import *
@property
def channel_name(self):
    return self._session._channel_name