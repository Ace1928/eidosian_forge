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
def parseFormattedText(text):
    """
    Parse text containing IRC formatting codes into structured information.

    Color codes are mapped from 0 to 15 and wrap around if greater than 15.

    @type text: C{str}
    @param text: Formatted text to parse.

    @return: Structured text and attributes.

    @since: 13.1
    """
    state = _FormattingParser()
    for ch in text:
        state.process(ch)
    return state.complete()