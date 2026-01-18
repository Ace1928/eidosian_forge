import copy
import operator
import re
import threading
def format_units_unicode(udict):
    res = format_units(udict)
    res = superscript(res)
    res = res.replace('**', '^').replace('*', '·')
    return res