import struct
import sys
def mkCrcFun(poly, initCrc=_CRC_INIT, rev=True, xorOut=0):
    """Return a function that computes the CRC using the specified polynomial.

    poly -- integer representation of the generator polynomial
    initCrc -- default initial CRC value
    rev -- when true, indicates that the data is processed bit reversed.
    xorOut -- the final XOR value

    The returned function has the following user interface
    def crcfun(data, crc=initCrc):
    """
    sizeBits, initCrc, xorOut = _verifyParams(poly, initCrc, xorOut)
    return _mkCrcFun(poly, sizeBits, initCrc, rev, xorOut)[0]