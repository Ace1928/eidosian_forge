import abc
import hmac
import hashlib
import math
@staticmethod
def createFor(messageVersion):
    from .hkdfv2 import HKDFv2
    from .hkdfv3 import HKDFv3
    if messageVersion == 2:
        return HKDFv2()
    elif messageVersion == 3:
        return HKDFv3()
    else:
        raise AssertionError('Unknown version: %s ' % messageVersion)