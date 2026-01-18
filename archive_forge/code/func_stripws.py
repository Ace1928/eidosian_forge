import tokenize
import string
def stripws(s):
    return ''.join((c for c in s if c not in string.whitespace))