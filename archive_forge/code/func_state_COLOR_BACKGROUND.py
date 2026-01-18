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
def state_COLOR_BACKGROUND(self, ch):
    """
        Handle the background color state.

        Background colors can consist of up to two digits and must occur after
        a foreground color and must be preceded by a I{,}. Any non-digit
        character is treated as invalid input and results in the state being
        set to "text".

        @param ch: The character being processed.
        """
    if ch.isdigit() and len(self._buffer) < 2:
        self._buffer += ch
    else:
        if self._buffer:
            col = int(self._buffer) % len(_IRC_COLORS)
            self.background = getattr(attributes.bg, _IRC_COLOR_NAMES[col])
            self._buffer = ''
        self.emit()
        self.state = 'TEXT'
        self.process(ch)