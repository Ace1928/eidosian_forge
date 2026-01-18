import sys
from typing import Optional, Sequence, Type
from zope.interface import Attribute, Interface
from twisted.plugin import getPlugins
from twisted.python import usage
def makeChecker(description):
    """
    Returns an L{twisted.cred.checkers.ICredentialsChecker} based on the
    contents of a descriptive string. Similar to
    L{twisted.application.strports}.
    """
    if ':' in description:
        authType, argstring = description.split(':', 1)
    else:
        authType = description
        argstring = ''
    return findCheckerFactory(authType).generateChecker(argstring)