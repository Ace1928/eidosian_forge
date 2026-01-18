from struct import pack, unpack, calcsize
def check_flags(val, fl):
    return val & fl == fl