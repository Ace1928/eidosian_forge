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
def handleCommand(self, command, prefix, params):
    """
        Determine the function to call for the given command and call it with
        the given arguments.

        @param command: The IRC command to determine the function for.
        @type command: L{bytes}

        @param prefix: The prefix of the IRC message (as returned by
            L{parsemsg}).
        @type prefix: L{bytes}

        @param params: A list of parameters to call the function with.
        @type params: L{list}
        """
    method = getattr(self, 'irc_%s' % command, None)
    try:
        if method is not None:
            method(prefix, params)
        else:
            self.irc_unknown(prefix, command, params)
    except BaseException:
        log.deferr()