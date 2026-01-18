from typing import Callable, Dict, Iterable, List, Tuple, Type, Union
from zope.interface import Interface, providedBy
from twisted.cred import error
from twisted.cred.checkers import ICredentialsChecker
from twisted.cred.credentials import ICredentials
from twisted.internet import defer
from twisted.internet.defer import Deferred, maybeDeferred
from twisted.python import failure, reflect
def listCredentialsInterfaces(self) -> List[Type[Interface]]:
    """
        Return list of credentials interfaces that can be used to login.
        """
    return list(self.checkers.keys())