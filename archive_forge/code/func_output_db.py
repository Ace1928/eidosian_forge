import functools
import re
import sys
from Xlib.support import lock
def output_db(prefix, db):
    res = ''
    for comp, group in db.items():
        if len(group) > 2:
            res = res + '%s%s: %s\n' % (prefix, comp, output_escape(group[2]))
        res = res + output_db(prefix + comp + '.', group[0])
        res = res + output_db(prefix + comp + '*', group[1])
    return res