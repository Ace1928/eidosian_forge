from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_feature_block_(self, variation=False):
    if variation:
        assert self.cur_token_ == 'variation'
    else:
        assert self.cur_token_ == 'feature'
    location = self.cur_token_location_
    tag = self.expect_tag_()
    vertical = tag in {'vkrn', 'vpal', 'vhal', 'valt'}
    stylisticset = None
    cv_feature = None
    size_feature = False
    if tag in self.SS_FEATURE_TAGS:
        stylisticset = tag
    elif tag in self.CV_FEATURE_TAGS:
        cv_feature = tag
    elif tag == 'size':
        size_feature = True
    if variation:
        conditionset = self.expect_name_()
    use_extension = False
    if self.next_token_ == 'useExtension':
        self.expect_keyword_('useExtension')
        use_extension = True
    if variation:
        block = self.ast.VariationBlock(tag, conditionset, use_extension=use_extension, location=location)
    else:
        block = self.ast.FeatureBlock(tag, use_extension=use_extension, location=location)
    self.parse_block_(block, vertical, stylisticset, size_feature, cv_feature)
    return block