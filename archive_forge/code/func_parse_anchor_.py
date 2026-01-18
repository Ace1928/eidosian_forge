from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_anchor_(self):
    self.expect_symbol_('<')
    self.expect_keyword_('anchor')
    location = self.cur_token_location_
    if self.next_token_ == 'NULL':
        self.expect_keyword_('NULL')
        self.expect_symbol_('>')
        return None
    if self.next_token_type_ == Lexer.NAME:
        name = self.expect_name_()
        anchordef = self.anchors_.resolve(name)
        if anchordef is None:
            raise FeatureLibError('Unknown anchor "%s"' % name, self.cur_token_location_)
        self.expect_symbol_('>')
        return self.ast.Anchor(anchordef.x, anchordef.y, name=name, contourpoint=anchordef.contourpoint, xDeviceTable=None, yDeviceTable=None, location=location)
    x, y = (self.expect_number_(variable=True), self.expect_number_(variable=True))
    contourpoint = None
    if self.next_token_ == 'contourpoint':
        self.expect_keyword_('contourpoint')
        contourpoint = self.expect_number_()
    if self.next_token_ == '<':
        xDeviceTable = self.parse_device_()
        yDeviceTable = self.parse_device_()
    else:
        xDeviceTable, yDeviceTable = (None, None)
    self.expect_symbol_('>')
    return self.ast.Anchor(x, y, name=None, contourpoint=contourpoint, xDeviceTable=xDeviceTable, yDeviceTable=yDeviceTable, location=location)