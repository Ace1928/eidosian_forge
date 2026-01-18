import binascii, codecs, zlib
from collections import OrderedDict
from reportlab.pdfbase import pdfutils
from reportlab import rl_config
from reportlab.lib.utils import open_for_read, makeFileName, isSeq, isBytes, isUnicode, _digester, isStr, bytestr, annotateException, TimeStamp
from reportlab.lib.rl_accel import escapePDF, fp_str, asciiBase85Encode, asciiBase85Decode
from reportlab.pdfbase import pdfmetrics
from hashlib import md5
from sys import stderr
import re
def maketree(self, document, destinationtree, Parent=None, toplevel=0):
    if toplevel:
        levelname = 'Outline'
        Parent = document.Reference(document.Outlines)
    else:
        self.count += 1
        levelname = 'Outline.%s' % self.count
        if Parent is None:
            raise ValueError('non-top level outline elt parent must be specified')
    if not isSeq(destinationtree):
        raise ValueError('destinationtree must be list or tuple, got %s')
    nelts = len(destinationtree)
    lastindex = nelts - 1
    lastelt = firstref = lastref = None
    destinationnamestotitles = self.destinationnamestotitles
    closedict = self.closedict
    for index in range(nelts):
        eltobj = OutlineEntryObject()
        eltobj.Parent = Parent
        eltname = '%s.%s' % (levelname, index)
        eltref = document.Reference(eltobj, eltname)
        if lastelt is not None:
            lastelt.Next = eltref
            eltobj.Prev = lastref
        if firstref is None:
            firstref = eltref
        lastref = eltref
        lastelt = eltobj
        lastref = eltref
        elt = destinationtree[index]
        if isinstance(elt, dict):
            leafdict = elt
        elif isinstance(elt, tuple):
            try:
                leafdict, subsections = elt
            except:
                raise ValueError('destination tree elt tuple should have two elts, got %s' % len(elt))
            eltobj.Count = count(subsections, closedict)
            eltobj.First, eltobj.Last = self.maketree(document, subsections, eltref)
        else:
            raise ValueError('destination tree elt should be dict or tuple, got %s' % type(elt))
        try:
            [(Title, Dest)] = list(leafdict.items())
        except:
            raise ValueError('bad outline leaf dictionary, should have one entry ' + bytestr(elt))
        eltobj.Title = destinationnamestotitles[Title]
        eltobj.Dest = Dest
        if isinstance(elt, tuple) and Dest in closedict:
            eltobj.Count = -eltobj.Count
    return (firstref, lastref)