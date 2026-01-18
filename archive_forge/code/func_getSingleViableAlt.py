from enum import Enum
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNConfig import ATNConfig
from antlr4.atn.ATNConfigSet import ATNConfigSet
from antlr4.atn.ATNState import RuleStopState
from antlr4.atn.SemanticContext import SemanticContext
@classmethod
def getSingleViableAlt(cls, altsets: list):
    viableAlts = set()
    for alts in altsets:
        minAlt = min(alts)
        viableAlts.add(minAlt)
        if len(viableAlts) > 1:
            return ATN.INVALID_ALT_NUMBER
    return min(viableAlts)