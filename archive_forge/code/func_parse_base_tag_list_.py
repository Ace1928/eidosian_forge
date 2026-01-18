from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_base_tag_list_(self):
    assert self.cur_token_ in ('HorizAxis.BaseTagList', 'VertAxis.BaseTagList'), self.cur_token_
    bases = []
    while self.next_token_ != ';':
        bases.append(self.expect_script_tag_())
    self.expect_symbol_(';')
    return bases