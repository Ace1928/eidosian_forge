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
def ctcpReply(self, user, channel, messages):
    """
        Dispatch method for any CTCP replies received.
        """
    for m in messages:
        method = getattr(self, 'ctcpReply_%s' % m[0], None)
        if method:
            method(user, channel, m[1])
        else:
            self.ctcpUnknownReply(user, channel, m[0], m[1])