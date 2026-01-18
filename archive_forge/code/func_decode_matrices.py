from hashlib import md5
import array
import re
def decode_matrices(byteseq):
    """
    Convert a sequence of 4n bytes into a list of n 2x2 integer matrices.
    """
    m = array.array('b')
    m.fromstring(byteseq)
    return [[list(m[n:n + 2]), list(m[n + 2:n + 4])] for n in range(0, len(m), 4)]