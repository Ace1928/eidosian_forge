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
def dccSend(self, user, file):
    """
        This is supposed to send a user a file directly.  This generally
        doesn't work on any client, and this method is included only for
        backwards compatibility and completeness.

        @param user: C{str} representing the user
        @param file: an open file (unknown, since this is not implemented)
        """
    raise NotImplementedError("XXX!!! Help!  I need to bind a socket, have it listen, and tell me its address.  (and stop accepting once we've made a single connection.)")