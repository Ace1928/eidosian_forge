from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_stat_name_(self):
    platEncID = None
    langID = None
    if self.next_token_type_ in Lexer.NUMBERS:
        platformID = self.expect_any_number_()
        location = self.cur_token_location_
        if platformID not in (1, 3):
            raise FeatureLibError('Expected platform id 1 or 3', location)
        if self.next_token_type_ in Lexer.NUMBERS:
            platEncID = self.expect_any_number_()
            langID = self.expect_any_number_()
    else:
        platformID = 3
        location = self.cur_token_location_
    if platformID == 1:
        platEncID = platEncID or 0
        langID = langID or 0
    else:
        platEncID = platEncID or 1
        langID = langID or 1033
    string = self.expect_string_()
    encoding = getEncoding(platformID, platEncID, langID)
    if encoding is None:
        raise FeatureLibError('Unsupported encoding', location)
    unescaped = self.unescape_string_(string, encoding)
    return (platformID, platEncID, langID, unescaped)