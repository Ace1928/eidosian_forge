from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_size_menuname_(self):
    assert self.is_cur_keyword_('sizemenuname')
    location = self.cur_token_location_
    platformID, platEncID, langID, string = self.parse_name_()
    return self.ast.FeatureNameStatement('size', platformID, platEncID, langID, string, location=location)