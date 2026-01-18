from antlr4 import *
from io import StringIO
import sys
def number_sempred(self, localctx: NumberContext, predIndex: int):
    if predIndex == 19:
        return not self.legacy_exponent_literal_as_decimal_enabled
    if predIndex == 20:
        return not self.legacy_exponent_literal_as_decimal_enabled
    if predIndex == 21:
        return self.legacy_exponent_literal_as_decimal_enabled