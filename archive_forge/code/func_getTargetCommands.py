from typing import Type
from twisted.internet import error
from twisted.internet.protocol import Protocol, connectionDone
from twisted.persisted import styles
from twisted.python.failure import Failure
from twisted.python.reflect import prefixedMethods
from twisted.words.im.locals import OFFLINE, OfflineError
def getTargetCommands(self, target):
    """finds group commands

        these commands are methods on me that start with imgroup_; they are
        called with a user present within this room as an argument

        you may want to override this in your group in order to filter for
        appropriate commands on the given user
        """
    return prefixedMethods(self, 'imtarget_')