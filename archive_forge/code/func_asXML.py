import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def asXML(self, doctag=None, namedItemsOnly=False, indent='', formatted=True):
    """Returns the parse results as XML. Tags are created for tokens and lists that have defined results names."""
    nl = '\n'
    out = []
    namedItems = dict(((v[1], k) for k, vlist in self.__tokdict.items() for v in vlist))
    nextLevelIndent = indent + '  '
    if not formatted:
        indent = ''
        nextLevelIndent = ''
        nl = ''
    selfTag = None
    if doctag is not None:
        selfTag = doctag
    elif self.__name:
        selfTag = self.__name
    if not selfTag:
        if namedItemsOnly:
            return ''
        else:
            selfTag = 'ITEM'
    out += [nl, indent, '<', selfTag, '>']
    worklist = self.__toklist
    for i, res in enumerate(worklist):
        if isinstance(res, ParseResults):
            if i in namedItems:
                out += [res.asXML(namedItems[i], namedItemsOnly and doctag is None, nextLevelIndent, formatted)]
            else:
                out += [res.asXML(None, namedItemsOnly and doctag is None, nextLevelIndent, formatted)]
        else:
            resTag = None
            if i in namedItems:
                resTag = namedItems[i]
            if not resTag:
                if namedItemsOnly:
                    continue
                else:
                    resTag = 'ITEM'
            xmlBodyText = _xml_escape(_ustr(res))
            out += [nl, nextLevelIndent, '<', resTag, '>', xmlBodyText, '</', resTag, '>']
    out += [nl, indent, '</', selfTag, '>']
    return ''.join(out)