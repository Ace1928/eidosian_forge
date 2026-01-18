from collections import OrderedDict
from itertools import chain
def name_sortfn(name):
    if name.count('_') > 1:
        return 2
    if 'CET' in name:
        return 1
    return 0