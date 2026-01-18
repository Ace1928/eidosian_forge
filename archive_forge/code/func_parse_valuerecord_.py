from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_valuerecord_(self, vertical):
    if self.next_token_type_ is Lexer.SYMBOL and self.next_token_ == '(' or self.next_token_type_ is Lexer.NUMBER:
        number, location = (self.expect_number_(variable=True), self.cur_token_location_)
        if vertical:
            val = self.ast.ValueRecord(yAdvance=number, vertical=vertical, location=location)
        else:
            val = self.ast.ValueRecord(xAdvance=number, vertical=vertical, location=location)
        return val
    self.expect_symbol_('<')
    location = self.cur_token_location_
    if self.next_token_type_ is Lexer.NAME:
        name = self.expect_name_()
        if name == 'NULL':
            self.expect_symbol_('>')
            return self.ast.ValueRecord()
        vrd = self.valuerecords_.resolve(name)
        if vrd is None:
            raise FeatureLibError('Unknown valueRecordDef "%s"' % name, self.cur_token_location_)
        value = vrd.value
        xPlacement, yPlacement = (value.xPlacement, value.yPlacement)
        xAdvance, yAdvance = (value.xAdvance, value.yAdvance)
    else:
        xPlacement, yPlacement, xAdvance, yAdvance = (self.expect_number_(variable=True), self.expect_number_(variable=True), self.expect_number_(variable=True), self.expect_number_(variable=True))
    if self.next_token_ == '<':
        xPlaDevice, yPlaDevice, xAdvDevice, yAdvDevice = (self.parse_device_(), self.parse_device_(), self.parse_device_(), self.parse_device_())
        allDeltas = sorted([delta for size, delta in (xPlaDevice if xPlaDevice else ()) + (yPlaDevice if yPlaDevice else ()) + (xAdvDevice if xAdvDevice else ()) + (yAdvDevice if yAdvDevice else ())])
        if allDeltas[0] < -128 or allDeltas[-1] > 127:
            raise FeatureLibError('Device value out of valid range (-128..127)', self.cur_token_location_)
    else:
        xPlaDevice, yPlaDevice, xAdvDevice, yAdvDevice = (None, None, None, None)
    self.expect_symbol_('>')
    return self.ast.ValueRecord(xPlacement, yPlacement, xAdvance, yAdvance, xPlaDevice, yPlaDevice, xAdvDevice, yAdvDevice, vertical=vertical, location=location)