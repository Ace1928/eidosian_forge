from antlr4 import *
from io import StringIO
import sys
def partitionSpec(self):
    return self.getTypedRuleContext(fugue_sqlParser.PartitionSpecContext, 0)