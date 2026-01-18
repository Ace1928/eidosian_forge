import datetime
import decimal
from typing import ClassVar, Dict, Type, TypeVar
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import address, defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.protocols import amp
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.test import iosim
from twisted.trial.unittest import TestCase
def tempSelfSigned():
    from twisted.internet import ssl
    sharedDN = ssl.DN(CN='shared')
    key = ssl.KeyPair.generate()
    cr = key.certificateRequest(sharedDN)
    sscrd = key.signCertificateRequest(sharedDN, cr, lambda dn: True, 1234567)
    cert = key.newCertificate(sscrd)
    return cert