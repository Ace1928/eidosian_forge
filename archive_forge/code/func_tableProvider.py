from antlr4 import *
from io import StringIO
import sys
def tableProvider(self):
    return self.getTypedRuleContext(fugue_sqlParser.TableProviderContext, 0)