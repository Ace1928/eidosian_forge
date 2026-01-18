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
def isupport_MAXLIST(self, params):
    """
        Maximum number of "list modes" a client may set on a channel at once.

        List modes are identified by the "addressModes" key in CHANMODES.
        """
    return self._splitParamArgs(params, _intOrDefault)