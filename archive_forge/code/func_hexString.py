from fontTools.misc.textTools import bytechr, bytesjoin, byteord
def hexString(s):
    import binascii
    return binascii.hexlify(s)