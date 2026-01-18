from antlr4 import *
from io import StringIO
import sys
def fugueCheckpointNamespace(self):
    return self.getTypedRuleContext(fugue_sqlParser.FugueCheckpointNamespaceContext, 0)