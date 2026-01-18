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
def irc_RPL_WELCOME(self, prefix, params):
    """
        Called when we have received the welcome from the server.
        """
    self.hostname = prefix
    self._registered = True
    self.nickname = self._attemptedNick
    self.signedOn()
    self.startHeartbeat()