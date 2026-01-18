from typing import Type
from twisted.internet import error
from twisted.internet.protocol import Protocol, connectionDone
from twisted.persisted import styles
from twisted.python.failure import Failure
from twisted.python.reflect import prefixedMethods
from twisted.words.im.locals import OFFLINE, OfflineError
def getGroupCommands(self):
    """finds group commands

        these commands are methods on me that start with imgroup_; they are
        called with no arguments
        """
    return prefixedMethods(self, 'imgroup_')