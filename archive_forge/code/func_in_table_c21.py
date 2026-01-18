from unicodedata import ucd_3_2_0 as unicodedata
def in_table_c21(code):
    return ord(code) < 128 and unicodedata.category(code) == 'Cc'