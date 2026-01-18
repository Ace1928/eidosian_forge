from unicodedata import ucd_3_2_0 as unicodedata
def in_table_d1(code):
    return unicodedata.bidirectional(code) in ('R', 'AL')