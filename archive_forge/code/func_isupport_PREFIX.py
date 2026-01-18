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
def isupport_PREFIX(self, params):
    """
        Mapping of channel modes that clients may have to status flags.
        """
    try:
        return self._parsePrefixParam(params[0])
    except ValueError:
        return self.getFeature('PREFIX')