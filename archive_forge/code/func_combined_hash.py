from hashlib import md5
import array
import re
def combined_hash(mfld):
    hash = str(' &and& '.join([basic_hash(mfld)] + cover_hash(mfld, (2, 3))))
    return hash.encode('utf8')