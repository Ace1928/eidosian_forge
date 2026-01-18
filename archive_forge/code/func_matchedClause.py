from antlr4 import *
from io import StringIO
import sys
def matchedClause(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.MatchedClauseContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.MatchedClauseContext, i)