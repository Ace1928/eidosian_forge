from antlr4 import *
from io import StringIO
import sys
def identifier_sempred(self, localctx: IdentifierContext, predIndex: int):
    if predIndex == 16:
        return not self.SQL_standard_keyword_behavior