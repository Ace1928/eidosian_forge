from antlr4 import *
from io import StringIO
import sys
def fugueOutputTransformTask(self):
    return self.getTypedRuleContext(fugue_sqlParser.FugueOutputTransformTaskContext, 0)