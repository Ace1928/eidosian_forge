from antlr4 import *
from io import StringIO
import sys
def strictIdentifier_sempred(self, localctx: StrictIdentifierContext, predIndex: int):
    if predIndex == 17:
        return self.SQL_standard_keyword_behavior
    if predIndex == 18:
        return not self.SQL_standard_keyword_behavior