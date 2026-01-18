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
def ctcpUnknownReply(self, user, channel, tag, data):
    """
        Called when a fitting ctcpReply_ method is not found.

        @param user: The hostmask of the user.
        @type user: L{bytes}

        @param channel: The name of the IRC channel.
        @type channel: L{bytes}

        @param tag: The CTCP request tag for which no fitting method is found.
        @type tag: L{bytes}

        @param data: The CTCP message.
        @type data: L{bytes}
        """
    log.msg(f'Unknown CTCP reply from {user}: {tag} {data}\n')