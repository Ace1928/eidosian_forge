from enum import Enum
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNConfig import ATNConfig
from antlr4.atn.ATNConfigSet import ATNConfigSet
from antlr4.atn.ATNState import RuleStopState
from antlr4.atn.SemanticContext import SemanticContext
@classmethod
def getUniqueAlt(cls, altsets: list):
    all = cls.getAlts(altsets)
    if len(all) == 1:
        return next(iter(all))
    return ATN.INVALID_ALT_NUMBER