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
def ctcpQuery_DCC(self, user, channel, data):
    """
        Initiate a Direct Client Connection

        @param user: The hostmask of the user/client.
        @type user: L{bytes}

        @param channel: The name of the IRC channel.
        @type channel: L{bytes}

        @param data: The DCC request message.
        @type data: L{bytes}
        """
    if not data:
        return
    dcctype = data.split(None, 1)[0].upper()
    handler = getattr(self, 'dcc_' + dcctype, None)
    if handler:
        if self.dcc_sessions is None:
            self.dcc_sessions = []
        data = data[len(dcctype) + 1:]
        handler(user, channel, data)
    else:
        nick = user.split('!')[0]
        self.ctcpMakeReply(nick, [('ERRMSG', f"DCC {data} :Unknown DCC type '{dcctype}'")])
        self.quirkyMessage(f'{user} offered unknown DCC type {dcctype}')