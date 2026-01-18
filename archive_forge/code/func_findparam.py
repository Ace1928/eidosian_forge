import os
import warnings
import re
def findparam(name, plist):
    name = name.lower() + '='
    n = len(name)
    for p in plist:
        if p[:n].lower() == name:
            return p[n:]
    return ''