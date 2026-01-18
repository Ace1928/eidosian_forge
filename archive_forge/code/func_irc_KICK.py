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
def irc_KICK(self, prefix, params):
    """
        Called when a user is kicked from a channel.
        """
    kicker = prefix.split('!')[0]
    channel = params[0]
    kicked = params[1]
    message = params[-1]
    if kicked.lower() == self.nickname.lower():
        self.kickedFrom(channel, kicker, message)
    else:
        self.userKicked(kicked, channel, kicker, message)