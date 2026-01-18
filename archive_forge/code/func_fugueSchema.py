from antlr4 import *
from io import StringIO
import sys
def fugueSchema(self):
    return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaContext, 0)