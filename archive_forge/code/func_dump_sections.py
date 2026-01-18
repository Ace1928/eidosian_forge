import datetime
import re
import sys
from decimal import Decimal
from toml.decoder import InlineTableDict
def dump_sections(self, o, sup):
    retstr = ''
    if sup != '' and sup[-1] != '.':
        sup += '.'
    retdict = self._dict()
    arraystr = ''
    for section in o:
        section = unicode(section)
        qsection = section
        if not re.match('^[A-Za-z0-9_-]+$', section):
            qsection = _dump_str(section)
        if not isinstance(o[section], dict):
            arrayoftables = False
            if isinstance(o[section], list):
                for a in o[section]:
                    if isinstance(a, dict):
                        arrayoftables = True
            if arrayoftables:
                for a in o[section]:
                    arraytabstr = '\n'
                    arraystr += '[[' + sup + qsection + ']]\n'
                    s, d = self.dump_sections(a, sup + qsection)
                    if s:
                        if s[0] == '[':
                            arraytabstr += s
                        else:
                            arraystr += s
                    while d:
                        newd = self._dict()
                        for dsec in d:
                            s1, d1 = self.dump_sections(d[dsec], sup + qsection + '.' + dsec)
                            if s1:
                                arraytabstr += '[' + sup + qsection + '.' + dsec + ']\n'
                                arraytabstr += s1
                            for s1 in d1:
                                newd[dsec + '.' + s1] = d1[s1]
                        d = newd
                    arraystr += arraytabstr
            elif o[section] is not None:
                retstr += qsection + ' = ' + unicode(self.dump_value(o[section])) + '\n'
        elif self.preserve and isinstance(o[section], InlineTableDict):
            retstr += qsection + ' = ' + self.dump_inline_table(o[section])
        else:
            retdict[qsection] = o[section]
    retstr += arraystr
    return (retstr, retdict)