import stringprep
from encodings import idna
from itertools import chain
from unicodedata import ucd_3_2_0 as unicodedata
from zope.interface import Interface, implementer
from incremental import Version
from twisted.python.deprecate import deprecatedModuleAttribute
def check_bidirectionals(self, string):
    found_LCat = False
    found_RandALCat = False
    for c in string:
        if stringprep.in_table_d1(c):
            found_RandALCat = True
        if stringprep.in_table_d2(c):
            found_LCat = True
    if found_LCat and found_RandALCat:
        raise UnicodeError('Violation of BIDI Requirement 2')
    if found_RandALCat and (not (stringprep.in_table_d1(string[0]) and stringprep.in_table_d1(string[-1]))):
        raise UnicodeError('Violation of BIDI Requirement 3')