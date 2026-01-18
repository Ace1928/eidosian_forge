import errno
import os
import socket
from unittest import skipIf
from twisted.internet import interfaces, reactor
from twisted.internet.defer import gatherResults, maybeDeferred
from twisted.internet.protocol import Protocol, ServerFactory
from twisted.internet.tcp import (
from twisted.python import log
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
def closeAll():
    for client in clients:
        client.close()