import os
from functools import partial
def find_token(tok, line):
    splitString = list(map(lambda x: x.strip(), line.split(tok)))
    if splitString is None or len(splitString) == 0 or len(splitString) == 1 or (splitString[0] == ''):
        return None
    else:
        return (splitString[0], tok.join(splitString[1:]))