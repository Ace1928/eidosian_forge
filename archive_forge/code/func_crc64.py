import binascii
def crc64(s):
    """Return the crc64 checksum for a sequence (string or Seq object).

    Note that the case is important:

    >>> crc64("ACGTACGTACGT")
    'CRC-C4FBB762C4A87EBD'
    >>> crc64("acgtACGTacgt")
    'CRC-DA4509DC64A87EBD'

    """
    crcl = 0
    crch = 0
    for c in s:
        shr = (crch & 255) << 24
        temp1h = crch >> 8
        temp1l = crcl >> 8 | shr
        idx = (crcl ^ ord(c)) & 255
        crch = temp1h ^ _table_h[idx]
        crcl = temp1l
    return f'CRC-{crch:08X}{crcl:08X}'