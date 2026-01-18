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
def ctcpQuery_SOURCE(self, user, channel, data):
    if data is not None:
        self.quirkyMessage(f"Why did {user} send '{data}' with a SOURCE query?")
    if self.sourceURL:
        nick = user.split('!')[0]
        self.ctcpMakeReply(nick, [('SOURCE', self.sourceURL), ('SOURCE', None)])