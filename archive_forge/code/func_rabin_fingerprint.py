import hashlib
def rabin_fingerprint(data):
    empty_64 = 13933329357911598997
    fp_table = []
    for i in range(256):
        fp = i
        for j in range(8):
            mask = -(fp & 1)
            fp = fp >> 1 ^ empty_64 & mask
        fp_table.append(fp)
    result = empty_64
    for byte in data:
        result = result >> 8 ^ fp_table[(result ^ byte) & 255]
    return result.to_bytes(length=8, byteorder='little', signed=False).hex()