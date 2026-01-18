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
def quirkyMessage(self, s):
    """
        This is called when I receive a message which is peculiar, but not
        wholly indecipherable.

        @param s: The peculiar message.
        @type s: L{bytes}
        """
    log.msg(s + '\n')