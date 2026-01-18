from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_glyphclass_definition_(self):
    location, name = (self.cur_token_location_, self.cur_token_)
    self.expect_symbol_('=')
    glyphs = self.parse_glyphclass_(accept_glyphname=False)
    self.expect_symbol_(';')
    glyphclass = self.ast.GlyphClassDefinition(name, glyphs, location=location)
    self.glyphclasses_.define(name, glyphclass)
    return glyphclass