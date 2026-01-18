from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_table_(self):
    assert self.is_cur_keyword_('table')
    location, name = (self.cur_token_location_, self.expect_tag_())
    table = self.ast.TableBlock(name, location=location)
    self.expect_symbol_('{')
    handler = {'GDEF': self.parse_table_GDEF_, 'head': self.parse_table_head_, 'hhea': self.parse_table_hhea_, 'vhea': self.parse_table_vhea_, 'name': self.parse_table_name_, 'BASE': self.parse_table_BASE_, 'OS/2': self.parse_table_OS_2_, 'STAT': self.parse_table_STAT_}.get(name)
    if handler:
        handler(table)
    else:
        raise FeatureLibError('"table %s" is not supported' % name.strip(), location)
    self.expect_symbol_('}')
    end_tag = self.expect_tag_()
    if end_tag != name:
        raise FeatureLibError('Expected "%s"' % name.strip(), self.cur_token_location_)
    self.expect_symbol_(';')
    return table