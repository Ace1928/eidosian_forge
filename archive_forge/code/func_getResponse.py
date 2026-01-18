import binascii
import os
import random
import time
from hashlib import md5
from zope.interface import Attribute, Interface, implementer
from twisted.python.compat import networkString
def getResponse(self, challenge):
    directives = self._parse(challenge)
    if b'rspauth' in directives:
        return b''
    charset = directives[b'charset'].decode('ascii')
    try:
        realm = directives[b'realm']
    except KeyError:
        realm = self.defaultRealm.encode(charset)
    return self._genResponse(charset, realm, directives[b'nonce'])