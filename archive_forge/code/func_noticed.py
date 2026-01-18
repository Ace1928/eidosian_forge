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
def noticed(self, user, channel, message):
    """
        Called when I have a notice from a user to me or a channel.

        If the client makes any automated replies, it must not do so in
        response to a NOTICE message, per the RFC::

            The difference between NOTICE and PRIVMSG is that
            automatic replies MUST NEVER be sent in response to a
            NOTICE message. [...] The object of this rule is to avoid
            loops between clients automatically sending something in
            response to something it received.
        """