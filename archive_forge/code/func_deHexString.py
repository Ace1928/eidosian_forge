from fontTools.misc.textTools import bytechr, bytesjoin, byteord
def deHexString(h):
    import binascii
    h = bytesjoin(h.split())
    return binascii.unhexlify(h)