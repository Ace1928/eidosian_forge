from __future__ import annotations
import sys
from collections import UserDict
from typing import TYPE_CHECKING, Union
from urllib.parse import quote as _quote, unquote as _unquote
from twisted.internet import defer, protocol
from twisted.protocols import basic, policies
from twisted.python import log
def sendCode(self, code, message=b''):
    """
        Send an SMTP-like code with a message.
        """
    self.sendLine(str(code).encode('ascii') + b' ' + message)