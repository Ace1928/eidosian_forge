from unicodedata import ucd_3_2_0 as unicodedata
def map_table_b2(a):
    al = map_table_b3(a)
    b = unicodedata.normalize('NFKC', al)
    bl = ''.join([map_table_b3(ch) for ch in b])
    c = unicodedata.normalize('NFKC', bl)
    if b != c:
        return c
    else:
        return al