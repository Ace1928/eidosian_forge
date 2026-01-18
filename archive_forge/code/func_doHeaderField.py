from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def doHeaderField(s, fld, attrmap):
    type, value = (fld.t[0], fld.t[1].replace('%5d', ']'))
    if not value:
        return
    if type == 'M':
        attrmap[type] = value
    elif type == 'L':
        try:
            s.unitL = lmap(int, fld.t[1].split('/'))
        except:
            info('illegal unit length:%s, 1/8 assumed' % fld.t[1])
            s.unitL = (1, 8)
        if len(s.unitL) == 1 or s.unitL[1] not in s.typeMap:
            info('L:%s is not allowed, 1/8 assumed' % fld.t[1])
            s.unitL = (1, 8)
    elif type == 'K':
        attrmap[type] = value
    elif type == 'T':
        s.title = s.title + '\n' + value if s.title else value
    elif type == 'U':
        sym = fld.t[2].strip('!+')
        s.usrSyms[value] = sym
    elif type == 'I':
        s.doField_I(type, value, lambda x, y, z: 0, lambda x: 0)
    elif type == 'Q':
        attrmap[type] = value
    elif type in 'CRZNOAGHBDFSP':
        type = s.metaMap.get(type, type)
        c = s.metadata.get(type, '')
        s.metadata[type] = c + '\n' + value if c else value
    else:
        info('skipped header: %s' % fld)