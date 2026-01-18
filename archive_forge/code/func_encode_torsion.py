from hashlib import md5
import array
import re
def encode_torsion(divisors):
    return ''.join((chr(x) for x in divisors)).encode('utf8')