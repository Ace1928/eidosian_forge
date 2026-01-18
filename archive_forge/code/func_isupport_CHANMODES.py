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
def isupport_CHANMODES(self, params):
    """
        Available channel modes.

        There are 4 categories of channel mode::

            addressModes - Modes that add or remove an address to or from a
            list, these modes always take a parameter.

            param - Modes that change a setting on a channel, these modes
            always take a parameter.

            setParam - Modes that change a setting on a channel, these modes
            only take a parameter when being set.

            noParam - Modes that change a setting on a channel, these modes
            never take a parameter.
        """
    try:
        return self._parseChanModesParam(params)
    except ValueError:
        return self.getFeature('CHANMODES')