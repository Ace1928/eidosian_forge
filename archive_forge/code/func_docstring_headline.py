import re
import sys
def docstring_headline(obj):
    if not obj.__doc__:
        return ''
    res = []
    for line in obj.__doc__.strip().splitlines():
        if line.strip():
            res.append(' ' + line.strip())
        else:
            break
    return ''.join(res).lstrip()