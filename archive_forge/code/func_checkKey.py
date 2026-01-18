import binascii
import errno
import sys
from base64 import decodebytes
from typing import IO, Any, Callable, Iterable, Iterator, Mapping, Optional, Tuple, cast
from zope.interface import Interface, implementer, providedBy
from incremental import Version
from typing_extensions import Literal, Protocol
from twisted.conch import error
from twisted.conch.ssh import keys
from twisted.cred.checkers import ICredentialsChecker
from twisted.cred.credentials import ISSHPrivateKey, IUsernamePassword
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet import defer
from twisted.logger import Logger
from twisted.plugins.cred_unix import verifyCryptedPassword
from twisted.python import failure, reflect
from twisted.python.deprecate import deprecatedModuleAttribute
from twisted.python.filepath import FilePath
from twisted.python.util import runAsEffectiveUser
def checkKey(self, credentials):
    """
        Retrieve files containing authorized keys and check against user
        credentials.
        """
    ouid, ogid = _lookupUser(self._userdb, credentials.username)[2:4]
    for filepath in self.getAuthorizedKeysFiles(credentials):
        if not filepath.exists():
            continue
        try:
            lines = filepath.open()
        except OSError as e:
            if e.errno == errno.EACCES:
                lines = runAsEffectiveUser(ouid, ogid, filepath.open)
            else:
                raise
        with lines:
            for l in lines:
                l2 = l.split()
                if len(l2) < 2:
                    continue
                try:
                    if decodebytes(l2[1]) == credentials.blob:
                        return True
                except binascii.Error:
                    continue
    return False