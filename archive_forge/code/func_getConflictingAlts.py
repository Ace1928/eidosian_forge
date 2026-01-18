from io import StringIO
from antlr4 import Parser, DFA
from antlr4.atn.ATNConfigSet import ATNConfigSet
from antlr4.error.ErrorListener import ErrorListener
def getConflictingAlts(self, reportedAlts: set, configs: ATNConfigSet):
    if reportedAlts is not None:
        return reportedAlts
    result = set()
    for config in configs:
        result.add(config.alt)
    return result