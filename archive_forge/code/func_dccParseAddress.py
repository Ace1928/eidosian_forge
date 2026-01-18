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
def dccParseAddress(address):
    if '.' in address:
        pass
    else:
        try:
            address = int(address)
        except ValueError:
            raise IRCBadMessage(f'Indecipherable address {address!r}')
        else:
            address = (address >> 24 & 255, address >> 16 & 255, address >> 8 & 255, address & 255)
            address = '.'.join(map(str, address))
    return address