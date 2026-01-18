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
def stripFormatting(text):
    """
    Remove all formatting codes from C{text}, leaving only the text.

    @type text: C{str}
    @param text: Formatted text to parse.

    @rtype: C{str}
    @return: Plain text without any control sequences.

    @since: 13.1
    """
    formatted = parseFormattedText(text)
    return _textattributes.flatten(formatted, _textattributes.DefaultFormattingState())