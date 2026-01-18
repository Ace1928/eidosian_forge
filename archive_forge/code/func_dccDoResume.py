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
def dccDoResume(self, user, file, port, resumePos):
    """
        Called when a client is trying to resume an offered file via DCC send.
        It should be either replied to with a DCC ACCEPT or ignored (default).

        @param user: The hostmask of the user who wants to resume the transfer
            of a file previously offered via DCC send.
        @type user: L{bytes}

        @param file: The name of the file to resume the transfer of.
        @type file: L{bytes}

        @param port: An integer representing the port of the requesting user.
        @type port: L{int}

        @param resumePos: The position in the file from where the transfer
            should resume.
        @type resumePos: L{int}
        """
    pass