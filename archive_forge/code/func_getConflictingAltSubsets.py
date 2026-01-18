from enum import Enum
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNConfig import ATNConfig
from antlr4.atn.ATNConfigSet import ATNConfigSet
from antlr4.atn.ATNState import RuleStopState
from antlr4.atn.SemanticContext import SemanticContext
@classmethod
def getConflictingAltSubsets(cls, configs: ATNConfigSet):
    configToAlts = dict()
    for c in configs:
        h = hash((c.state.stateNumber, c.context))
        alts = configToAlts.get(h, None)
        if alts is None:
            alts = set()
            configToAlts[h] = alts
        alts.add(c.alt)
    return configToAlts.values()