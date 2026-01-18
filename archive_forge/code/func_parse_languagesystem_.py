from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_languagesystem_(self):
    assert self.cur_token_ == 'languagesystem'
    location = self.cur_token_location_
    script = self.expect_script_tag_()
    language = self.expect_language_tag_()
    self.expect_symbol_(';')
    return self.ast.LanguageSystemStatement(script, language, location=location)