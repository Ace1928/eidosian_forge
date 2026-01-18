import re
def dwarr2inttuple(dwarr):
    fields = dwarr.type.fields()
    lo, hi = fields[0].type.range()
    return tuple([int(dwarr[x]) for x in range(lo, hi + 1)])