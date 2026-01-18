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
def irc_RPL_TOPIC(self, prefix, params):
    """
        Called when the topic for a channel is initially reported or when it
        subsequently changes.
        """
    user = prefix.split('!')[0]
    channel = params[1]
    newtopic = params[2]
    self.topicUpdated(user, channel, newtopic)