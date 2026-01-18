from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_base_script_record_(self, count):
    script_tag = self.expect_script_tag_()
    base_tag = self.expect_script_tag_()
    coords = [self.expect_number_() for i in range(count)]
    return (script_tag, base_tag, coords)