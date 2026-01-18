import abc
import hmac
import hashlib
import math
def deriveSecrets(self, inputKeyMaterial, info, outputLength, salt=None):
    salt = salt or bytearray(self.__class__.HASH_OUTPUT_SIZE)
    prk = self.extract(salt, inputKeyMaterial)
    return self.expand(prk, info, outputLength)