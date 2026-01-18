from unicodedata import ucd_3_2_0 as unicodedata
def in_table_c4(code):
    c = ord(code)
    if c < 64976:
        return False
    if c < 65008:
        return True
    return ord(code) & 65535 in (65534, 65535)