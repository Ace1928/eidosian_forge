@staticmethod
def intToByteArray(_bytes, offset, value):
    _bytes[offset + 3] = value % 256
    _bytes[offset + 2] = (value >> 8) % 256
    _bytes[offset + 1] = (value >> 16) % 256
    _bytes[offset] = (value >> 24) % 256
    return 4