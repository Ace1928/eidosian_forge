import socket
import time
import warnings
from collections import OrderedDict
from typing import Dict, List
from zope.interface import Interface, implementer
from twisted import cred
from twisted.internet import defer, protocol, reactor
from twisted.protocols import basic
from twisted.python import log
def messageDone(self, remainingData=''):
    assert self.state == 'body'
    self.message.creationFinished()
    self.messageReceived(self.message)
    self.reset(remainingData)