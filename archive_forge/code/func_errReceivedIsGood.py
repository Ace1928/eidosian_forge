import sys
import warnings
from functools import wraps
from io import BytesIO
from twisted.internet import defer, protocol
from twisted.python import failure
def errReceivedIsGood(self, text):
    self.s.write(text)