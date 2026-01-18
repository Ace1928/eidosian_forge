from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_nameid_(self):
    assert self.cur_token_ == 'nameid', self.cur_token_
    location, nameID = (self.cur_token_location_, self.expect_any_number_())
    if nameID > 32767:
        raise FeatureLibError('Name id value cannot be greater than 32767', self.cur_token_location_)
    platformID, platEncID, langID, string = self.parse_name_()
    return self.ast.NameRecord(nameID, platformID, platEncID, langID, string, location=location)