import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def getmultilineblock(rout, blockname, comment=1, counter=0):
    try:
        r = rout['f2pyenhancements'].get(blockname)
    except KeyError:
        return
    if not r:
        return
    if counter > 0 and isinstance(r, str):
        return
    if isinstance(r, list):
        if counter >= len(r):
            return
        r = r[counter]
    if r[:3] == "'''":
        if comment:
            r = '\t/* start ' + blockname + ' multiline (' + repr(counter) + ') */\n' + r[3:]
        else:
            r = r[3:]
        if r[-3:] == "'''":
            if comment:
                r = r[:-3] + '\n\t/* end multiline (' + repr(counter) + ')*/'
            else:
                r = r[:-3]
        else:
            errmess("%s multiline block should end with `'''`: %s\n" % (blockname, repr(r)))
    return r