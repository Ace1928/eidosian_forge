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
def sendLine(self, line):
    if self.lineRate is None:
        self._reallySendLine(line)
    else:
        self._queue.append(line)
        if not self._queueEmptying:
            self._sendLine()