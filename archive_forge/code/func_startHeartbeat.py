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
def startHeartbeat(self):
    """
        Start sending I{PING} messages every L{IRCClient.heartbeatInterval}
        seconds to keep the connection to the server alive during periods of no
        activity.

        @since: 11.1
        """
    self.stopHeartbeat()
    if self.heartbeatInterval is None:
        return
    self._heartbeat = self._createHeartbeat()
    self._heartbeat.start(self.heartbeatInterval, now=False)