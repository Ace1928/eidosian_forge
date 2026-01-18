import abc
import hmac
import hashlib
import math
def extract(self, salt, inputKeyMaterial):
    mac = hmac.new(bytes(salt), digestmod=hashlib.sha256)
    mac.update(bytes(inputKeyMaterial))
    return mac.digest()