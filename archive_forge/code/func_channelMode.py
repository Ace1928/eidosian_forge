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
def channelMode(self, user, channel, mode, *args):
    """
        Send information about the mode of a channel.

        @type user: C{str} or C{unicode}
        @param user: The user receiving the name list.  Only their nickname,
            not the full hostmask.

        @type channel: C{str} or C{unicode}
        @param channel: The channel for which this is the namelist.

        @type mode: C{str}
        @param mode: A string describing this channel's modes.

        @param args: Any additional arguments required by the modes.
        """
    self.sendLine(':%s %s %s %s %s %s' % (self.hostname, RPL_CHANNELMODEIS, user, channel, mode, ' '.join(args)))