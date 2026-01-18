from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_table_STAT_(self, table):
    statements = table.statements
    design_axes = []
    while self.next_token_ != '}' or self.cur_comments_:
        self.advance_lexer_(comments=True)
        if self.cur_token_type_ is Lexer.COMMENT:
            statements.append(self.ast.Comment(self.cur_token_, location=self.cur_token_location_))
        elif self.cur_token_type_ is Lexer.NAME:
            if self.is_cur_keyword_('ElidedFallbackName'):
                names = self.parse_STAT_ElidedFallbackName()
                statements.append(self.ast.ElidedFallbackName(names))
            elif self.is_cur_keyword_('ElidedFallbackNameID'):
                value = self.expect_number_()
                statements.append(self.ast.ElidedFallbackNameID(value))
                self.expect_symbol_(';')
            elif self.is_cur_keyword_('DesignAxis'):
                designAxis = self.parse_STAT_design_axis()
                design_axes.append(designAxis.tag)
                statements.append(designAxis)
                self.expect_symbol_(';')
            elif self.is_cur_keyword_('AxisValue'):
                axisValueRecord = self.parse_STAT_axis_value_()
                for location in axisValueRecord.locations:
                    if location.tag not in design_axes:
                        raise FeatureLibError(f'DesignAxis not defined for {location.tag}.', self.cur_token_location_)
                statements.append(axisValueRecord)
                self.expect_symbol_(';')
            else:
                raise FeatureLibError(f'Unexpected token {self.cur_token_}', self.cur_token_location_)
        elif self.cur_token_ == ';':
            continue