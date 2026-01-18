from io import StringIO
from functools import reduce
from antlr4.PredictionContext import PredictionContext, merge
from antlr4.Utils import str_list
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNConfig import ATNConfig
from antlr4.atn.SemanticContext import SemanticContext
from antlr4.error.Errors import UnsupportedOperationException, IllegalStateException
def getPredicates(self):
    return list((cfg.semanticContext for cfg in self.configs if cfg.semanticContext != SemanticContext.NONE))