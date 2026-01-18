from hashlib import md5
import array
import re
def cover_hash(mfld, degrees):
    return [repr(sorted([(cover_type(C), C.homology()) for C in mfld.covers(degree, method='snappea')])) for degree in degrees]