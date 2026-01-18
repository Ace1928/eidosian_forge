import logging
import re
def compare_identifiers(a, b):
    anum = NUMERIC.search(a)
    bnum = NUMERIC.search(b)
    if anum and bnum:
        a = int(a)
        b = int(b)
    if anum and (not bnum):
        return -1
    elif bnum and (not anum):
        return 1
    elif a < b:
        return -1
    elif a > b:
        return 1
    else:
        return 0