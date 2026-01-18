from pyasn1 import error
@property
def defaultType(self):
    """Return default ASN.1 type being returned for any missing *TagSet*"""
    return self.__defaultType