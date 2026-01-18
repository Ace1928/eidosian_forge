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
def ctcpExtract(message):
    """
    Extract CTCP data from a string.

    @return: A C{dict} containing two keys:
       - C{'extended'}: A list of CTCP (tag, data) tuples.
       - C{'normal'}: A list of strings which were not inside a CTCP delimiter.
    """
    extended_messages = []
    normal_messages = []
    retval = {'extended': extended_messages, 'normal': normal_messages}
    messages = message.split(X_DELIM)
    odd = 0
    while messages:
        if odd:
            extended_messages.append(messages.pop(0))
        else:
            normal_messages.append(messages.pop(0))
        odd = not odd
    extended_messages[:] = list(filter(None, extended_messages))
    normal_messages[:] = list(filter(None, normal_messages))
    extended_messages[:] = list(map(ctcpDequote, extended_messages))
    for i in range(len(extended_messages)):
        m = extended_messages[i].split(SPC, 1)
        tag = m[0]
        if len(m) > 1:
            data = m[1]
        else:
            data = None
        extended_messages[i] = (tag, data)
    return retval