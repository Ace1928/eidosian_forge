import base64
import hmac
import random
import re
import time
from binascii import hexlify
from hashlib import md5
from zope.interface import Interface, implementer
from twisted.cred import error
from twisted.cred._digest import calcHA1, calcHA2, calcResponse
from twisted.python.compat import nativeString, networkString
from twisted.python.deprecate import deprecatedModuleAttribute
from twisted.python.randbytes import secureRandom
from twisted.python.versions import Version
def checkHash(self, digestHash):
    """
        Verify that the credentials represented by this object agree with the
        credentials represented by the I{H(A1)} given in C{digestHash}.

        @param digestHash: A precomputed H(A1) value based on the username,
            realm, and password associate with this credentials object.
        """
    response = self.fields.get('response')
    uri = self.fields.get('uri')
    nonce = self.fields.get('nonce')
    cnonce = self.fields.get('cnonce')
    nc = self.fields.get('nc')
    algo = self.fields.get('algorithm', b'md5').lower()
    qop = self.fields.get('qop', b'auth')
    expected = calcResponse(calcHA1(algo, None, None, None, nonce, cnonce, preHA1=digestHash), calcHA2(algo, self.method, uri, qop, None), algo, nonce, nc, cnonce, qop)
    return expected == response