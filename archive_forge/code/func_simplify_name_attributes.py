from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
def simplify_name_attributes(pid, eid, lid):
    if pid == 3 and eid == 1 and (lid == 1033):
        return ''
    elif pid == 1 and eid == 0 and (lid == 0):
        return '1'
    else:
        return '{} {} {}'.format(pid, eid, lid)