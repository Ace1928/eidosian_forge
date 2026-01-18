from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_markClass_(self):
    assert self.is_cur_keyword_('markClass')
    location = self.cur_token_location_
    glyphs = self.parse_glyphclass_(accept_glyphname=True)
    if not glyphs.glyphSet():
        raise FeatureLibError('Empty glyph class in mark class definition', location)
    anchor = self.parse_anchor_()
    name = self.expect_class_name_()
    self.expect_symbol_(';')
    markClass = self.doc_.markClasses.get(name)
    if markClass is None:
        markClass = self.ast.MarkClass(name)
        self.doc_.markClasses[name] = markClass
        self.glyphclasses_.define(name, markClass)
    mcdef = self.ast.MarkClassDefinition(markClass, anchor, glyphs, location=location)
    markClass.addDefinition(mcdef)
    return mcdef