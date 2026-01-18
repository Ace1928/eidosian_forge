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
def say(self, channel, message, length=None):
    """
        Send a message to a channel

        @type channel: C{str}
        @param channel: The channel to say the message on. If it has no prefix,
            C{'#'} will be prepended to it.
        @type message: C{str}
        @param message: The message to say.
        @type length: C{int}
        @param length: The maximum number of octets to send at a time.  This has
            the effect of turning a single call to C{msg()} into multiple
            commands to the server.  This is useful when long messages may be
            sent that would otherwise cause the server to kick us off or
            silently truncate the text we are sending.  If None is passed, the
            entire message is always send in one command.
        """
    if channel[0] not in CHANNEL_PREFIXES:
        channel = '#' + channel
    self.msg(channel, message, length)