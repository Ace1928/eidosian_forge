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
def ctcpQuery(self, user, channel, messages):
    """
        Dispatch method for any CTCP queries received.

        Duplicated CTCP queries are ignored and no dispatch is
        made. Unrecognized CTCP queries invoke L{IRCClient.ctcpUnknownQuery}.
        """
    seen = set()
    for tag, data in messages:
        method = getattr(self, 'ctcpQuery_%s' % tag, None)
        if tag not in seen:
            if method is not None:
                method(user, channel, data)
            else:
                self.ctcpUnknownQuery(user, channel, tag, data)
        seen.add(tag)