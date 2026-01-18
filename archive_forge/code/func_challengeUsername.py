import hashlib
import hmac
from zope.interface import implementer
from twisted.cred import credentials
from twisted.mail._except import IllegalClientResponse
from twisted.mail.interfaces import IChallengeResponse, IClientAuthentication
from twisted.python.compat import nativeString
def challengeUsername(self, secret, chal):
    self.challengeResponse = self.challengeSecret
    return self.user