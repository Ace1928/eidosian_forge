import re
import string
def encode_integer_list(L):
    return ''.join(map(encode_int, L))