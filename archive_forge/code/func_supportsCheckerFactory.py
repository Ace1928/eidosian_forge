import sys
from typing import Optional, Sequence, Type
from zope.interface import Attribute, Interface
from twisted.plugin import getPlugins
from twisted.python import usage
def supportsCheckerFactory(self, factory):
    """
        Returns whether a checker factory will provide at least one of
        the credentials interfaces that we care about.
        """
    for interface in factory.credentialInterfaces:
        if self.supportsInterface(interface):
            return True
    return False