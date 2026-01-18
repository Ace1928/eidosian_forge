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
def state_TEXT(self, ch):
    """
        Handle the "text" state.

        Along with regular text, single token formatting codes are handled
        in this state too.

        @param ch: The character being processed.
        """
    formatName = self._formatCodes.get(ch)
    if formatName == 'color':
        self.emit()
        self.state = 'COLOR_FOREGROUND'
    elif formatName is None:
        self._buffer += ch
    else:
        self.emit()
        if formatName == 'off':
            self._attrs = set()
            self.foreground = self.background = None
        else:
            self._attrs.symmetric_difference_update([formatName])