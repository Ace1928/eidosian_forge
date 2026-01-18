import hashlib
import hmac
from .sendermessagekey import SenderMessageKey
def getDerivative(self, seed, key):
    mac = hmac.new(bytes(key), bytes(seed), digestmod=hashlib.sha256)
    return mac.digest()