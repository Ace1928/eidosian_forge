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
def dccDescribe(data):
    """
    Given the data chunk from a DCC query, return a descriptive string.

    @param data: The data from a DCC query.
    @type data: L{bytes}

    @rtype: L{bytes}
    @return: A descriptive string.
    """
    orig_data = data
    data = data.split()
    if len(data) < 4:
        return orig_data
    dcctype, arg, address, port = data[:4]
    if '.' in address:
        pass
    else:
        try:
            address = int(address)
        except ValueError:
            pass
        else:
            address = (address >> 24 & 255, address >> 16 & 255, address >> 8 & 255, address & 255)
            address = '.'.join(map(str, address))
    if dcctype == 'SEND':
        filename = arg
        size_txt = ''
        if len(data) >= 5:
            try:
                size = int(data[4])
                size_txt = ' of size %d bytes' % (size,)
            except ValueError:
                pass
        dcc_text = "SEND for file '{}'{} at host {}, port {}".format(filename, size_txt, address, port)
    elif dcctype == 'CHAT':
        dcc_text = f'CHAT for host {address}, port {port}'
    else:
        dcc_text = orig_data
    return dcc_text