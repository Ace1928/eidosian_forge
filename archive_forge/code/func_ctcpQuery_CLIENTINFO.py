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
def ctcpQuery_CLIENTINFO(self, user, channel, data):
    """
        A master index of what CTCP tags this client knows.

        If no arguments are provided, respond with a list of known tags, sorted
        in alphabetical order.
        If an argument is provided, provide human-readable help on
        the usage of that tag.
        """
    nick = user.split('!')[0]
    if not data:
        names = sorted(reflect.prefixedMethodNames(self.__class__, 'ctcpQuery_'))
        self.ctcpMakeReply(nick, [('CLIENTINFO', ' '.join(names))])
    else:
        args = data.split()
        method = getattr(self, f'ctcpQuery_{args[0]}', None)
        if not method:
            self.ctcpMakeReply(nick, [('ERRMSG', "CLIENTINFO %s :Unknown query '%s'" % (data, args[0]))])
            return
        doc = getattr(method, '__doc__', '')
        self.ctcpMakeReply(nick, [('CLIENTINFO', doc)])