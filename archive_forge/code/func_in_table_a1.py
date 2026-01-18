from unicodedata import ucd_3_2_0 as unicodedata
def in_table_a1(code):
    if unicodedata.category(code) != 'Cn':
        return False
    c = ord(code)
    if 64976 <= c < 65008:
        return False
    return c & 65535 not in (65534, 65535)