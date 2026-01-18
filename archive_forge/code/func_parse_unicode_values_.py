import fontTools.voltLib.ast as ast
from fontTools.voltLib.lexer import Lexer
from fontTools.voltLib.error import VoltLibError
from io import open
def parse_unicode_values_(self):
    location = self.cur_token_location_
    try:
        unicode_values = self.expect_string_().split(',')
        unicode_values = [int(uni[2:], 16) for uni in unicode_values if uni != '']
    except ValueError as err:
        raise VoltLibError(str(err), location)
    return unicode_values if unicode_values != [] else None