import errno
import operator
import os
import random
import re
import shlex
import socket
import stat
import string
import struct
import sys
import textwrap
import time
import traceback
from functools import reduce
from os import path
from typing import Optional
from twisted.internet import protocol, reactor, task
from twisted.persisted import styles
from twisted.protocols import basic
from twisted.python import _textattributes, log, reflect
def dcc_SEND(self, user, channel, data):
    data = shlex.split(data)
    if len(data) < 3:
        raise IRCBadMessage(f'malformed DCC SEND request: {data!r}')
    filename, address, port = data[:3]
    address = dccParseAddress(address)
    try:
        port = int(port)
    except ValueError:
        raise IRCBadMessage(f'Indecipherable port {port!r}')
    size = -1
    if len(data) >= 4:
        try:
            size = int(data[3])
        except ValueError:
            pass
    self.dccDoSend(user, address, port, filename, size, data)