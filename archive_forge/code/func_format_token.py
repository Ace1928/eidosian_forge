import sys
import tokenize
def format_token(t):
    r = repr(t)
    if r.startswith('TokenInfo('):
        r = r[10:]
    pos = r.find("), line='")
    if pos < 0:
        pos = r.find('), line="')
    if pos > 0:
        r = r[:pos + 1]
    return r