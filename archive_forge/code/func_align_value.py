from struct import pack, unpack, calcsize
def align_value(val, b):
    return val + -val % b