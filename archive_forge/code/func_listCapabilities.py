import base64
import binascii
import warnings
from hashlib import md5
from typing import Optional
from zope.interface import implementer
from twisted import cred
from twisted.internet import defer, interfaces, task
from twisted.mail import smtp
from twisted.mail._except import POP3ClientError, POP3Error, _POP3MessageDeleted
from twisted.mail.interfaces import (
from twisted.protocols import basic, policies
from twisted.python import log
from twisted.mail._except import (
from twisted.mail._pop3client import POP3Client as AdvancedPOP3Client
def listCapabilities(self):
    """
        Return a list of server capabilities suitable for use in a CAPA
        response.

        @rtype: L{list} of L{bytes}
        @return: A list of server capabilities.
        """
    baseCaps = [b'TOP', b'USER', b'UIDL', b'PIPELINE', b'CELERITY', b'AUSPEX', b'POTENCE']
    if IServerFactory.providedBy(self.factory):
        try:
            v = self.factory.cap_IMPLEMENTATION()
            if v and (not isinstance(v, bytes)):
                v = str(v).encode('utf-8')
        except NotImplementedError:
            pass
        except BaseException:
            log.err()
        else:
            baseCaps.append(b'IMPLEMENTATION ' + v)
        try:
            v = self.factory.cap_EXPIRE()
            if v and (not isinstance(v, bytes)):
                v = str(v).encode('utf-8')
        except NotImplementedError:
            pass
        except BaseException:
            log.err()
        else:
            if v is None:
                v = b'NEVER'
            if self.factory.perUserExpiration():
                if self.mbox:
                    v = str(self.mbox.messageExpiration).encode('utf-8')
                else:
                    v = v + b' USER'
            baseCaps.append(b'EXPIRE ' + v)
        try:
            v = self.factory.cap_LOGIN_DELAY()
            if v and (not isinstance(v, bytes)):
                v = str(v).encode('utf-8')
        except NotImplementedError:
            pass
        except BaseException:
            log.err()
        else:
            if self.factory.perUserLoginDelay():
                if self.mbox:
                    v = str(self.mbox.loginDelay).encode('utf-8')
                else:
                    v = v + b' USER'
            baseCaps.append(b'LOGIN-DELAY ' + v)
        try:
            v = self.factory.challengers
        except AttributeError:
            pass
        except BaseException:
            log.err()
        else:
            baseCaps.append(b'SASL ' + b' '.join(v.keys()))
    return baseCaps