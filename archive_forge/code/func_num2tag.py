import struct, warnings
def num2tag(n):
    if n < 2097152:
        return str(n)
    else:
        return struct.unpack('4s', struct.pack('>L', n))[0].replace(b'\x00', b'').decode()