from antlr4 import *
from io import StringIO
import sys
def bucketSpec(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.BucketSpecContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.BucketSpecContext, i)