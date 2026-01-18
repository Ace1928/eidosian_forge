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
def notice(self, user, message, length=None):
    """
        Send a notice to a user.

        Notices are like normal message, but should never get automated
        replies.

        @type user: C{str}
        @param user: The user to send a notice to.

        @type message: C{str}
        @param message: The contents of the notice to send.

        @param length: Maximum number of octets to send in a single
            command, including the IRC protocol framing. If L{None} is given
            then L{IRCClient._safeMaximumLineLength} is used to determine a
            value.
        @type length: C{int}
        """
    self._sendMessage('NOTICE', user, message, length)