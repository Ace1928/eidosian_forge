from enum import Enum
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNConfig import ATNConfig
from antlr4.atn.ATNConfigSet import ATNConfigSet
from antlr4.atn.ATNState import RuleStopState
from antlr4.atn.SemanticContext import SemanticContext
@classmethod
def getStateToAltMap(cls, configs: ATNConfigSet):
    m = dict()
    for c in configs:
        alts = m.get(c.state, None)
        if alts is None:
            alts = set()
            m[c.state] = alts
        alts.add(c.alt)
    return m