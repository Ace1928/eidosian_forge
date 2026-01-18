import reportlab
def calcChecksum(data):
    """Calculates TTF-style checksums"""
    data = rawBytes(data)
    if len(data) & 3:
        data = data + (4 - (len(data) & 3)) * b'\x00'
    return sum(unpack('>%dl' % (len(data) >> 2), data)) & 4294967295