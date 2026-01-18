import stringprep
from encodings import idna
from itertools import chain
from unicodedata import ucd_3_2_0 as unicodedata
from zope.interface import Interface, implementer
from incremental import Version
from twisted.python.deprecate import deprecatedModuleAttribute
def check_unassigneds(self, string):
    for c in string:
        if stringprep.in_table_a1(c):
            raise UnicodeError('Unassigned code point %s' % repr(c))