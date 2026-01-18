import os
import signal
from testtools.helpers import try_import
from testtools import skipIf
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
def make_reactor(self):
    from twisted.internet import reactor
    return reactor