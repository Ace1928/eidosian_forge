from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def expect_markClass_reference_(self):
    name = self.expect_class_name_()
    mc = self.glyphclasses_.resolve(name)
    if mc is None:
        raise FeatureLibError('Unknown markClass @%s' % name, self.cur_token_location_)
    if not isinstance(mc, self.ast.MarkClass):
        raise FeatureLibError('@%s is not a markClass' % name, self.cur_token_location_)
    return mc