import errno
import os
import sys
from twisted.python.util import untilConcludes
from twisted.trial import unittest
import os, errno
def eintrWrite(fd, data):
    err = OSError()
    err.errno = errno.EINTR
    raise err