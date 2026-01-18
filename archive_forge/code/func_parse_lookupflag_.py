from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_lookupflag_(self):
    assert self.is_cur_keyword_('lookupflag')
    location = self.cur_token_location_
    if self.next_token_type_ == Lexer.NUMBER:
        value = self.expect_number_()
        self.expect_symbol_(';')
        return self.ast.LookupFlagStatement(value, location=location)
    value_seen = False
    value, markAttachment, markFilteringSet = (0, None, None)
    flags = {'RightToLeft': 1, 'IgnoreBaseGlyphs': 2, 'IgnoreLigatures': 4, 'IgnoreMarks': 8}
    seen = set()
    while self.next_token_ != ';':
        if self.next_token_ in seen:
            raise FeatureLibError('%s can be specified only once' % self.next_token_, self.next_token_location_)
        seen.add(self.next_token_)
        if self.next_token_ == 'MarkAttachmentType':
            self.expect_keyword_('MarkAttachmentType')
            markAttachment = self.parse_glyphclass_(accept_glyphname=False)
        elif self.next_token_ == 'UseMarkFilteringSet':
            self.expect_keyword_('UseMarkFilteringSet')
            markFilteringSet = self.parse_glyphclass_(accept_glyphname=False)
        elif self.next_token_ in flags:
            value_seen = True
            value = value | flags[self.expect_name_()]
        else:
            raise FeatureLibError('"%s" is not a recognized lookupflag' % self.next_token_, self.next_token_location_)
    self.expect_symbol_(';')
    if not any([value_seen, markAttachment, markFilteringSet]):
        raise FeatureLibError('lookupflag must have a value', self.next_token_location_)
    return self.ast.LookupFlagStatement(value, markAttachment=markAttachment, markFilteringSet=markFilteringSet, location=location)